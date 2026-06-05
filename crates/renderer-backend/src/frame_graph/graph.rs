//! Core Frame Graph Data Structure for wgpu 25.x Render Pass Scheduling
//!
//! This module provides the primary frame graph abstraction for organizing
//! render and compute passes with automatic dependency resolution and
//! resource lifetime management.
//!
//! # Architecture
//!
//! The frame graph consists of two node types:
//! - **PassNode**: Represents a render, compute, transfer, or ray-tracing pass
//! - **ResourceNode**: Represents a GPU resource (buffer or texture)
//!
//! Dependencies are expressed through resource usage: passes declare which
//! resources they read from and write to. The graph compiler performs:
//! 1. Topological sorting to determine execution order
//! 2. Hazard detection (RAW, WAW, WAR)
//! 3. Resource lifetime analysis for aliasing opportunities
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::frame_graph::graph::*;
//!
//! let mut graph = FrameGraph::new();
//!
//! // Create resources
//! let color = graph.add_resource("color_buffer", ResourceType::Texture2D, ResourceLifetime::Transient);
//! let depth = graph.add_resource("depth_buffer", ResourceType::Texture2D, ResourceLifetime::Transient);
//!
//! // Create passes
//! let shadow_pass = graph.add_pass("shadow_pass", PassType::Render);
//! let main_pass = graph.add_pass("main_pass", PassType::Render);
//!
//! // Connect passes to resources
//! graph.connect(shadow_pass, depth, ResourceAccess::Write);
//! graph.connect(main_pass, depth, ResourceAccess::Read);
//! graph.connect(main_pass, color, ResourceAccess::Write);
//!
//! // Compile and execute
//! graph.compile()?;
//! graph.execute(&mut render_context)?;
//! ```

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use crate::resource_state::PipelineStage;

// ---------------------------------------------------------------------------
// ID Types
// ---------------------------------------------------------------------------

/// Unique identifier for frame graph resources.
///
/// Uses u64 to allow for large resource counts and potential
/// cross-frame resource tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ResourceId(pub u64);

impl ResourceId {
    /// Invalid/null resource ID.
    pub const INVALID: Self = Self(u64::MAX);

    /// Creates a new resource ID from a raw value.
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    #[inline]
    pub const fn raw(&self) -> u64 {
        self.0
    }

    /// Returns true if this is the invalid/null ID.
    #[inline]
    pub const fn is_invalid(&self) -> bool {
        self.0 == u64::MAX
    }
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_invalid() {
            write!(f, "ResourceId::INVALID")
        } else {
            write!(f, "ResourceId({})", self.0)
        }
    }
}

impl Default for ResourceId {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Unique identifier for frame graph passes.
///
/// Uses u64 for consistency with ResourceId and to allow for
/// large pass counts in complex frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct PassId(pub u64);

impl PassId {
    /// Invalid/null pass ID.
    pub const INVALID: Self = Self(u64::MAX);

    /// Creates a new pass ID from a raw value.
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw ID value.
    #[inline]
    pub const fn raw(&self) -> u64 {
        self.0
    }

    /// Returns true if this is the invalid/null ID.
    #[inline]
    pub const fn is_invalid(&self) -> bool {
        self.0 == u64::MAX
    }
}

impl fmt::Display for PassId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_invalid() {
            write!(f, "PassId::INVALID")
        } else {
            write!(f, "PassId({})", self.0)
        }
    }
}

impl Default for PassId {
    fn default() -> Self {
        Self::INVALID
    }
}

// ---------------------------------------------------------------------------
// Resource Access
// ---------------------------------------------------------------------------

/// How a pass accesses a resource.
///
/// Used for dependency tracking and hazard detection:
/// - RAW (Read After Write): A read depends on a prior write
/// - WAW (Write After Write): A write depends on a prior write
/// - WAR (Write After Read): A write depends on a prior read
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceAccess {
    /// The pass only reads from the resource.
    Read,
    /// The pass only writes to the resource.
    Write,
    /// The pass both reads and writes the resource.
    ReadWrite,
}

impl ResourceAccess {
    /// Returns true if this access involves reading.
    #[inline]
    pub const fn is_read(&self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    /// Returns true if this access involves writing.
    #[inline]
    pub const fn is_write(&self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }

    /// Determines if this access conflicts with another access.
    ///
    /// Conflicts occur when:
    /// - RAW: self reads, other writes
    /// - WAW: self writes, other writes
    /// - WAR: self writes, other reads
    ///
    /// Returns true if there is a potential hazard requiring synchronization.
    #[inline]
    pub const fn conflicts_with(&self, other: &Self) -> bool {
        // RAW: we read, they wrote
        // WAW: we write, they wrote
        // WAR: we write, they read
        let self_reads = self.is_read();
        let self_writes = self.is_write();
        let other_reads = other.is_read();
        let other_writes = other.is_write();

        // RAW or WAW or WAR
        (self_reads && other_writes) || (self_writes && other_writes) || (self_writes && other_reads)
    }
}

impl fmt::Display for ResourceAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read => write!(f, "Read"),
            Self::Write => write!(f, "Write"),
            Self::ReadWrite => write!(f, "ReadWrite"),
        }
    }
}

impl Default for ResourceAccess {
    fn default() -> Self {
        Self::Read
    }
}

// ---------------------------------------------------------------------------
// Resource Usage
// ---------------------------------------------------------------------------

/// Describes how a pass uses a specific resource at a specific pipeline stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceUsage {
    /// The resource being accessed.
    pub resource: ResourceId,
    /// How the resource is accessed.
    pub access: ResourceAccess,
    /// The pipeline stage where the access occurs.
    pub stage: PipelineStage,
}

impl ResourceUsage {
    /// Creates a new resource usage entry.
    pub fn new(resource: ResourceId, access: ResourceAccess, stage: PipelineStage) -> Self {
        Self {
            resource,
            access,
            stage,
        }
    }

    /// Creates a read usage at the fragment shader stage.
    pub fn read(resource: ResourceId) -> Self {
        Self::new(resource, ResourceAccess::Read, PipelineStage::FragmentShader)
    }

    /// Creates a write usage at the color output stage.
    pub fn write(resource: ResourceId) -> Self {
        Self::new(resource, ResourceAccess::Write, PipelineStage::ColorOutput)
    }

    /// Creates a read-write usage at the compute shader stage.
    pub fn read_write(resource: ResourceId) -> Self {
        Self::new(resource, ResourceAccess::ReadWrite, PipelineStage::ComputeShader)
    }
}

impl fmt::Display for ResourceUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResourceUsage({} {} at {:?})", self.resource, self.access, self.stage)
    }
}

// ---------------------------------------------------------------------------
// Pass Type
// ---------------------------------------------------------------------------

/// The type of GPU workload a pass represents.
///
/// Determines scheduling rules and which wgpu command encoder methods
/// are used to record the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassType {
    /// A rasterization pass using vertex and fragment shaders.
    Render,
    /// A compute shader dispatch.
    Compute,
    /// A GPU copy/transfer operation.
    Transfer,
    /// A ray-tracing dispatch.
    RayTracing,
}

impl PassType {
    /// Returns true if this is a graphics (rasterization) pass.
    #[inline]
    pub const fn is_graphics(&self) -> bool {
        matches!(self, Self::Render)
    }

    /// Returns true if this is a compute pass.
    #[inline]
    pub const fn is_compute(&self) -> bool {
        matches!(self, Self::Compute)
    }

    /// Returns true if this is a transfer pass.
    #[inline]
    pub const fn is_transfer(&self) -> bool {
        matches!(self, Self::Transfer)
    }

    /// Returns true if this is a ray-tracing pass.
    #[inline]
    pub const fn is_raytracing(&self) -> bool {
        matches!(self, Self::RayTracing)
    }
}

impl fmt::Display for PassType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Render => write!(f, "Render"),
            Self::Compute => write!(f, "Compute"),
            Self::Transfer => write!(f, "Transfer"),
            Self::RayTracing => write!(f, "RayTracing"),
        }
    }
}

impl Default for PassType {
    fn default() -> Self {
        Self::Render
    }
}

// ---------------------------------------------------------------------------
// Resource Type
// ---------------------------------------------------------------------------

/// The physical type of a GPU resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// A linear GPU buffer.
    Buffer,
    /// A 2D texture.
    Texture2D,
    /// A 3D / volume texture.
    Texture3D,
    /// A cube map texture.
    TextureCube,
    /// A 2D texture array.
    Texture2DArray,
    /// An acceleration structure for ray tracing.
    AccelerationStructure,
}

impl ResourceType {
    /// Returns true if this is a buffer type.
    #[inline]
    pub const fn is_buffer(&self) -> bool {
        matches!(self, Self::Buffer)
    }

    /// Returns true if this is a texture type.
    #[inline]
    pub const fn is_texture(&self) -> bool {
        matches!(
            self,
            Self::Texture2D | Self::Texture3D | Self::TextureCube | Self::Texture2DArray
        )
    }

    /// Returns true if this is an acceleration structure.
    #[inline]
    pub const fn is_acceleration_structure(&self) -> bool {
        matches!(self, Self::AccelerationStructure)
    }
}

impl fmt::Display for ResourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Buffer => write!(f, "Buffer"),
            Self::Texture2D => write!(f, "Texture2D"),
            Self::Texture3D => write!(f, "Texture3D"),
            Self::TextureCube => write!(f, "TextureCube"),
            Self::Texture2DArray => write!(f, "Texture2DArray"),
            Self::AccelerationStructure => write!(f, "AccelerationStructure"),
        }
    }
}

impl Default for ResourceType {
    fn default() -> Self {
        Self::Texture2D
    }
}

// ---------------------------------------------------------------------------
// Resource Lifetime
// ---------------------------------------------------------------------------

/// Lifetime category for frame graph resources.
///
/// Determines how resources are allocated and whether they can be aliased.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GraphResourceLifetime {
    /// Resource is allocated per-frame and destroyed at frame end.
    /// Transient resources may be aliased with non-overlapping resources.
    Transient,
    /// Resource persists across multiple frames.
    /// Cannot be aliased.
    Persistent,
    /// Resource is provided externally (e.g., swapchain image).
    /// The frame graph tracks state but does not manage allocation.
    Imported,
}

impl GraphResourceLifetime {
    /// Returns true if this is a transient resource.
    #[inline]
    pub const fn is_transient(&self) -> bool {
        matches!(self, Self::Transient)
    }

    /// Returns true if this is a persistent resource.
    #[inline]
    pub const fn is_persistent(&self) -> bool {
        matches!(self, Self::Persistent)
    }

    /// Returns true if this is an imported resource.
    #[inline]
    pub const fn is_imported(&self) -> bool {
        matches!(self, Self::Imported)
    }

    /// Returns true if this resource can be aliased.
    #[inline]
    pub const fn can_alias(&self) -> bool {
        matches!(self, Self::Transient)
    }
}

impl fmt::Display for GraphResourceLifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient => write!(f, "Transient"),
            Self::Persistent => write!(f, "Persistent"),
            Self::Imported => write!(f, "Imported"),
        }
    }
}

impl Default for GraphResourceLifetime {
    fn default() -> Self {
        Self::Transient
    }
}

// ---------------------------------------------------------------------------
// Render Context
// ---------------------------------------------------------------------------

/// Execution context passed to pass callbacks during frame graph execution.
///
/// This is a placeholder for the actual wgpu rendering context that would
/// contain command encoders, device references, and other execution state.
#[derive(Clone, Debug, Default)]
pub struct RenderContext {
    /// Current frame index (monotonically increasing).
    pub frame_index: u64,
    /// Debug label for the current pass.
    pub current_pass_label: String,
}

impl RenderContext {
    /// Creates a new render context for a given frame.
    pub fn new(frame_index: u64) -> Self {
        Self {
            frame_index,
            current_pass_label: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Pass Node
// ---------------------------------------------------------------------------

/// Callback type for pass execution.
/// Using a type alias for clarity.
pub type PassCallback = Box<dyn FnOnce(&mut RenderContext) + Send + Sync>;

/// A pass node in the frame graph.
///
/// Represents a single GPU workload (render pass, compute dispatch, etc.)
/// with its resource dependencies.
pub struct PassNode {
    /// Unique identifier for this pass.
    pub id: PassId,
    /// Human-readable name for debugging.
    pub name: String,
    /// The type of GPU workload.
    pub pass_type: PassType,
    /// Resources this pass reads from.
    pub inputs: Vec<ResourceUsage>,
    /// Resources this pass writes to.
    pub outputs: Vec<ResourceUsage>,
    /// Optional execution callback.
    callback: Option<PassCallback>,
    /// Whether this pass is enabled (disabled passes are skipped).
    pub enabled: bool,
}

impl PassNode {
    /// Creates a new pass node.
    pub fn new(id: PassId, name: impl Into<String>, pass_type: PassType) -> Self {
        Self {
            id,
            name: name.into(),
            pass_type,
            inputs: Vec::new(),
            outputs: Vec::new(),
            callback: None,
            enabled: true,
        }
    }

    /// Adds an input resource usage (resource the pass reads).
    pub fn add_input(&mut self, usage: ResourceUsage) {
        self.inputs.push(usage);
    }

    /// Adds an input resource with default read settings.
    pub fn add_input_resource(&mut self, resource: ResourceId, stage: PipelineStage) {
        self.inputs.push(ResourceUsage::new(resource, ResourceAccess::Read, stage));
    }

    /// Adds an output resource usage (resource the pass writes).
    pub fn add_output(&mut self, usage: ResourceUsage) {
        self.outputs.push(usage);
    }

    /// Adds an output resource with default write settings.
    pub fn add_output_resource(&mut self, resource: ResourceId, stage: PipelineStage) {
        self.outputs.push(ResourceUsage::new(resource, ResourceAccess::Write, stage));
    }

    /// Sets the execution callback for this pass.
    pub fn set_callback<F>(&mut self, callback: F)
    where
        F: FnOnce(&mut RenderContext) + Send + Sync + 'static,
    {
        self.callback = Some(Box::new(callback));
    }

    /// Takes the callback out of this pass (for execution).
    pub fn take_callback(&mut self) -> Option<PassCallback> {
        self.callback.take()
    }

    /// Returns true if this pass has a callback set.
    pub fn has_callback(&self) -> bool {
        self.callback.is_some()
    }

    /// Returns all resources this pass accesses (both inputs and outputs).
    pub fn all_resources(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.inputs.iter().map(|u| u.resource).chain(self.outputs.iter().map(|u| u.resource))
    }

    /// Returns the set of resource IDs this pass reads.
    pub fn read_resources(&self) -> Vec<ResourceId> {
        self.inputs.iter().filter(|u| u.access.is_read()).map(|u| u.resource).collect()
    }

    /// Returns the set of resource IDs this pass writes.
    pub fn write_resources(&self) -> Vec<ResourceId> {
        self.outputs.iter().filter(|u| u.access.is_write()).map(|u| u.resource).collect()
    }
}

impl fmt::Debug for PassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PassNode")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("pass_type", &self.pass_type)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("has_callback", &self.callback.is_some())
            .field("enabled", &self.enabled)
            .finish()
    }
}

impl fmt::Display for PassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PassNode({} \"{}\", type={}, inputs={}, outputs={}, enabled={})",
            self.id,
            self.name,
            self.pass_type,
            self.inputs.len(),
            self.outputs.len(),
            self.enabled
        )
    }
}

// ---------------------------------------------------------------------------
// Resource Node
// ---------------------------------------------------------------------------

/// A resource node in the frame graph.
///
/// Represents a GPU resource with tracking of which passes produce
/// and consume it.
#[derive(Clone, Debug)]
pub struct ResourceNode {
    /// Unique identifier for this resource.
    pub id: ResourceId,
    /// Human-readable name for debugging.
    pub name: String,
    /// The physical type of the resource.
    pub resource_type: ResourceType,
    /// The pass that produces/creates this resource (if any).
    pub producer: Option<PassId>,
    /// The passes that consume/read this resource.
    pub consumers: Vec<PassId>,
    /// The lifetime category of this resource.
    pub lifetime: GraphResourceLifetime,
}

impl ResourceNode {
    /// Creates a new resource node.
    pub fn new(
        id: ResourceId,
        name: impl Into<String>,
        resource_type: ResourceType,
        lifetime: GraphResourceLifetime,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            resource_type,
            producer: None,
            consumers: Vec::new(),
            lifetime,
        }
    }

    /// Returns true if this resource is transient (frame-local).
    #[inline]
    pub fn is_transient(&self) -> bool {
        self.lifetime.is_transient()
    }

    /// Returns true if this resource is imported (externally managed).
    #[inline]
    pub fn is_imported(&self) -> bool {
        self.lifetime.is_imported()
    }

    /// Returns true if this resource is persistent (survives across frames).
    #[inline]
    pub fn is_persistent(&self) -> bool {
        self.lifetime.is_persistent()
    }

    /// Sets the producer pass for this resource.
    pub fn set_producer(&mut self, pass: PassId) {
        self.producer = Some(pass);
    }

    /// Adds a consumer pass for this resource.
    pub fn add_consumer(&mut self, pass: PassId) {
        if !self.consumers.contains(&pass) {
            self.consumers.push(pass);
        }
    }

    /// Returns the number of passes that reference this resource.
    pub fn reference_count(&self) -> usize {
        let producer_count = if self.producer.is_some() { 1 } else { 0 };
        producer_count + self.consumers.len()
    }
}

impl fmt::Display for ResourceNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ResourceNode({} \"{}\", type={}, lifetime={}, producer={:?}, consumers={})",
            self.id,
            self.name,
            self.resource_type,
            self.lifetime,
            self.producer,
            self.consumers.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Frame Graph Error
// ---------------------------------------------------------------------------

/// Errors that can occur during frame graph compilation or execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameGraphError {
    /// The graph contains a cyclic dependency.
    CyclicDependency,
    /// A referenced resource does not exist.
    MissingResource(ResourceId),
    /// A referenced pass does not exist.
    MissingPass(PassId),
    /// Invalid resource access pattern.
    InvalidAccess(String),
    /// Graph has not been compiled.
    NotCompiled,
    /// A pass callback failed during execution.
    ExecutionFailed(String),
}

impl fmt::Display for FrameGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CyclicDependency => write!(f, "Cyclic dependency detected in frame graph"),
            Self::MissingResource(id) => write!(f, "Missing resource: {}", id),
            Self::MissingPass(id) => write!(f, "Missing pass: {}", id),
            Self::InvalidAccess(msg) => write!(f, "Invalid resource access: {}", msg),
            Self::NotCompiled => write!(f, "Frame graph has not been compiled"),
            Self::ExecutionFailed(msg) => write!(f, "Pass execution failed: {}", msg),
        }
    }
}

impl std::error::Error for FrameGraphError {}

// ---------------------------------------------------------------------------
// Frame Graph
// ---------------------------------------------------------------------------

/// The main frame graph structure for organizing GPU workloads.
///
/// The frame graph provides automatic dependency resolution, resource
/// lifetime tracking, and execution ordering for GPU passes.
pub struct FrameGraph {
    /// All passes in the graph, keyed by ID.
    passes: HashMap<PassId, PassNode>,
    /// All resources in the graph, keyed by ID.
    resources: HashMap<ResourceId, ResourceNode>,
    /// Topologically sorted execution order (valid after compile).
    execution_order: Vec<PassId>,
    /// Counter for generating unique pass IDs.
    next_pass_id: u64,
    /// Counter for generating unique resource IDs.
    next_resource_id: u64,
    /// Whether the graph has been compiled.
    compiled: bool,
}

impl FrameGraph {
    /// Creates a new empty frame graph.
    pub fn new() -> Self {
        Self {
            passes: HashMap::new(),
            resources: HashMap::new(),
            execution_order: Vec::new(),
            next_pass_id: 0,
            next_resource_id: 0,
            compiled: false,
        }
    }

    /// Adds a new pass to the graph.
    ///
    /// Returns the unique ID of the new pass.
    pub fn add_pass(&mut self, name: impl Into<String>, pass_type: PassType) -> PassId {
        let id = PassId::new(self.next_pass_id);
        self.next_pass_id += 1;
        let pass = PassNode::new(id, name, pass_type);
        self.passes.insert(id, pass);
        self.compiled = false;
        id
    }

    /// Adds a new resource to the graph.
    ///
    /// Returns the unique ID of the new resource.
    pub fn add_resource(
        &mut self,
        name: impl Into<String>,
        resource_type: ResourceType,
        lifetime: GraphResourceLifetime,
    ) -> ResourceId {
        let id = ResourceId::new(self.next_resource_id);
        self.next_resource_id += 1;
        let resource = ResourceNode::new(id, name, resource_type, lifetime);
        self.resources.insert(id, resource);
        self.compiled = false;
        id
    }

    /// Gets a reference to a pass by ID.
    pub fn get_pass(&self, id: PassId) -> Option<&PassNode> {
        self.passes.get(&id)
    }

    /// Gets a mutable reference to a pass by ID.
    pub fn get_pass_mut(&mut self, id: PassId) -> Option<&mut PassNode> {
        self.compiled = false;
        self.passes.get_mut(&id)
    }

    /// Gets a reference to a resource by ID.
    pub fn get_resource(&self, id: ResourceId) -> Option<&ResourceNode> {
        self.resources.get(&id)
    }

    /// Gets a mutable reference to a resource by ID.
    pub fn get_resource_mut(&mut self, id: ResourceId) -> Option<&mut ResourceNode> {
        self.compiled = false;
        self.resources.get_mut(&id)
    }

    /// Connects a pass to a resource with a specific access type.
    ///
    /// This creates a dependency edge in the graph:
    /// - Read access: adds the resource as an input to the pass
    /// - Write access: adds the resource as an output and sets the pass as producer
    /// - ReadWrite: does both
    pub fn connect(&mut self, pass: PassId, resource: ResourceId, access: ResourceAccess) {
        // Determine the pipeline stage based on pass type
        let stage = if let Some(p) = self.passes.get(&pass) {
            match p.pass_type {
                PassType::Render if access.is_write() => PipelineStage::ColorOutput,
                PassType::Render => PipelineStage::FragmentShader,
                PassType::Compute => PipelineStage::ComputeShader,
                PassType::Transfer => PipelineStage::Transfer,
                PassType::RayTracing => PipelineStage::ComputeShader, // closest approximation
            }
        } else {
            PipelineStage::AllCommands
        };

        self.connect_with_stage(pass, resource, access, stage);
    }

    /// Connects a pass to a resource with explicit pipeline stage.
    pub fn connect_with_stage(
        &mut self,
        pass: PassId,
        resource: ResourceId,
        access: ResourceAccess,
        stage: PipelineStage,
    ) {
        let usage = ResourceUsage::new(resource, access, stage);

        // Update pass
        if let Some(p) = self.passes.get_mut(&pass) {
            if access.is_read() {
                p.inputs.push(usage.clone());
            }
            if access.is_write() {
                p.outputs.push(usage);
            }
        }

        // Update resource
        if let Some(r) = self.resources.get_mut(&resource) {
            if access.is_write() {
                r.set_producer(pass);
            }
            if access.is_read() {
                r.add_consumer(pass);
            }
        }

        self.compiled = false;
    }

    /// Compiles the frame graph, resolving dependencies and determining execution order.
    ///
    /// This performs:
    /// 1. Validation of resource references
    /// 2. Topological sorting of passes
    /// 3. Cycle detection
    pub fn compile(&mut self) -> Result<(), FrameGraphError> {
        // Validate all resource references exist
        for pass in self.passes.values() {
            for usage in pass.inputs.iter().chain(pass.outputs.iter()) {
                if !self.resources.contains_key(&usage.resource) {
                    return Err(FrameGraphError::MissingResource(usage.resource));
                }
            }
        }

        // Build adjacency list for topological sort
        // Edge: pass A -> pass B if A produces a resource that B reads
        let mut in_degree: HashMap<PassId, usize> = HashMap::new();
        let mut adj: HashMap<PassId, Vec<PassId>> = HashMap::new();

        for pass_id in self.passes.keys() {
            in_degree.insert(*pass_id, 0);
            adj.insert(*pass_id, Vec::new());
        }

        // Build edges based on resource dependencies
        for (resource_id, resource) in &self.resources {
            if let Some(producer) = resource.producer {
                for &consumer in &resource.consumers {
                    if producer != consumer {
                        // Producer pass -> Consumer pass edge
                        if let Some(edges) = adj.get_mut(&producer) {
                            if !edges.contains(&consumer) {
                                edges.push(consumer);
                                *in_degree.entry(consumer).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        // Kahn's algorithm for topological sort
        let mut queue: Vec<PassId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut order = Vec::with_capacity(self.passes.len());

        while let Some(pass_id) = queue.pop() {
            // Skip disabled passes
            if let Some(pass) = self.passes.get(&pass_id) {
                if !pass.enabled {
                    continue;
                }
            }

            order.push(pass_id);

            if let Some(neighbors) = adj.get(&pass_id) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(&neighbor) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push(neighbor);
                        }
                    }
                }
            }
        }

        // Check for cycles (not all enabled passes are in the order)
        let enabled_count = self.passes.values().filter(|p| p.enabled).count();
        if order.len() != enabled_count {
            return Err(FrameGraphError::CyclicDependency);
        }

        self.execution_order = order;
        self.compiled = true;
        Ok(())
    }

    /// Executes the compiled frame graph.
    ///
    /// Iterates through passes in topological order, invoking each pass's callback.
    pub fn execute(&mut self, context: &mut RenderContext) -> Result<(), FrameGraphError> {
        if !self.compiled {
            return Err(FrameGraphError::NotCompiled);
        }

        // Take execution order to avoid borrow issues
        let order = std::mem::take(&mut self.execution_order);

        for pass_id in &order {
            if let Some(pass) = self.passes.get_mut(pass_id) {
                if pass.enabled {
                    context.current_pass_label = pass.name.clone();
                    if let Some(callback) = pass.take_callback() {
                        callback(context);
                    }
                }
            }
        }

        // Restore execution order
        self.execution_order = order;
        Ok(())
    }

    /// Resets the frame graph for the next frame.
    ///
    /// This clears the execution order but preserves passes and resources.
    /// Call this at the start of each frame before building the new graph.
    pub fn reset(&mut self) {
        self.execution_order.clear();
        self.compiled = false;

        // Clear consumer lists (they're rebuilt each frame)
        for resource in self.resources.values_mut() {
            resource.consumers.clear();
            resource.producer = None;
        }

        // Clear pass inputs/outputs
        for pass in self.passes.values_mut() {
            pass.inputs.clear();
            pass.outputs.clear();
            pass.callback = None;
        }
    }

    /// Clears the entire frame graph.
    pub fn clear(&mut self) {
        self.passes.clear();
        self.resources.clear();
        self.execution_order.clear();
        self.next_pass_id = 0;
        self.next_resource_id = 0;
        self.compiled = false;
    }

    /// Returns the number of passes in the graph.
    #[inline]
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Returns the number of resources in the graph.
    #[inline]
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Returns true if the graph has been compiled.
    #[inline]
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Returns the execution order (valid only after compile).
    pub fn execution_order(&self) -> &[PassId] {
        &self.execution_order
    }

    /// Returns an iterator over all passes.
    pub fn passes(&self) -> impl Iterator<Item = &PassNode> {
        self.passes.values()
    }

    /// Returns an iterator over all resources.
    pub fn resources(&self) -> impl Iterator<Item = &ResourceNode> {
        self.resources.values()
    }

    /// Finds passes that write to a specific resource.
    pub fn find_writers(&self, resource: ResourceId) -> Vec<PassId> {
        self.passes
            .values()
            .filter(|p| p.outputs.iter().any(|u| u.resource == resource))
            .map(|p| p.id)
            .collect()
    }

    /// Finds passes that read from a specific resource.
    pub fn find_readers(&self, resource: ResourceId) -> Vec<PassId> {
        self.passes
            .values()
            .filter(|p| p.inputs.iter().any(|u| u.resource == resource))
            .map(|p| p.id)
            .collect()
    }

    /// Validates the graph structure without compiling.
    pub fn validate(&self) -> Result<(), FrameGraphError> {
        // Check all resource references
        for pass in self.passes.values() {
            for usage in pass.inputs.iter().chain(pass.outputs.iter()) {
                if !self.resources.contains_key(&usage.resource) {
                    return Err(FrameGraphError::MissingResource(usage.resource));
                }
            }
        }
        Ok(())
    }

    /// Adds a render pass to the frame graph with full configuration.
    ///
    /// This method creates a pass node based on the provided render pass
    /// configuration and automatically sets up resource dependencies based
    /// on the attachments.
    ///
    /// # Arguments
    ///
    /// * `config` - The render pass configuration
    /// * `executor` - The pass executor that will record draw commands
    ///
    /// # Returns
    ///
    /// The PassId of the newly created pass.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::frame_graph::passes::*;
    /// use renderer_backend::frame_graph::graph::*;
    ///
    /// let mut graph = FrameGraph::new();
    /// let color = graph.add_resource("color", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    ///
    /// let config = RenderPassConfig::with_color(
    ///     "main_pass",
    ///     ColorAttachment::new(color),
    /// );
    ///
    /// let pass_id = graph.add_render_pass(config, NoOpExecutor);
    /// ```
    pub fn add_render_pass<E>(&mut self, config: super::passes::RenderPassConfig, executor: E) -> PassId
    where
        E: super::passes::PassExecutor + 'static,
    {
        // Create the pass node
        let pass_id = self.add_pass(&config.name, PassType::Render);

        // Connect color attachment resources
        for attachment in &config.color_attachments {
            // Load operation means we read from the resource
            if attachment.load_op.is_load() {
                self.connect(pass_id, attachment.resource, ResourceAccess::Read);
            }
            // Store operation means we write to the resource
            if attachment.store_op.is_store() {
                self.connect(pass_id, attachment.resource, ResourceAccess::Write);
            }
            // Resolve target is always written
            if let Some(resolve) = attachment.resolve_target {
                self.connect(pass_id, resolve, ResourceAccess::Write);
            }
        }

        // Connect depth attachment resource
        if let Some(depth) = &config.depth_attachment {
            // Read-only depth or load operation
            if depth.read_only || depth.depth_load_op.is_load() {
                self.connect(pass_id, depth.resource, ResourceAccess::Read);
            }
            // Writes to depth or stencil
            if depth.writes_depth() || depth.writes_stencil() {
                self.connect(pass_id, depth.resource, ResourceAccess::Write);
            }
        }

        pass_id
    }

    /// Adds a compute pass to the frame graph with full configuration.
    ///
    /// This method creates a pass node based on the provided compute pass
    /// configuration and automatically sets up resource dependencies based
    /// on the builder's read/write tracking.
    ///
    /// # Arguments
    ///
    /// * `config` - The compute pass configuration
    /// * `reads` - Resources this pass reads from
    /// * `writes` - Resources this pass writes to
    /// * `executor` - The pass executor that will record compute commands
    ///
    /// # Returns
    ///
    /// The PassId of the newly created pass.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::frame_graph::passes::*;
    /// use renderer_backend::frame_graph::graph::*;
    ///
    /// let mut graph = FrameGraph::new();
    /// let input = graph.add_resource("input", ResourceType::Buffer, GraphResourceLifetime::Transient);
    /// let output = graph.add_resource("output", ResourceType::Buffer, GraphResourceLifetime::Transient);
    ///
    /// let (config, reads, writes) = ComputePassBuilder::new("compute_pass")
    ///     .dispatch(256, 1, 1)
    ///     .read_resource(input)
    ///     .write_resource(output)
    ///     .build_with_deps();
    ///
    /// let pass_id = graph.add_compute_pass(config, reads, writes, NoOpComputeExecutor);
    /// ```
    pub fn add_compute_pass<E>(
        &mut self,
        config: super::passes::ComputePassConfig,
        reads: Vec<ResourceId>,
        writes: Vec<ResourceId>,
        executor: E,
    ) -> PassId
    where
        E: super::passes::ComputePassExecutor + 'static,
    {
        // Create the pass node
        let pass_id = self.add_pass(&config.name, PassType::Compute);

        // Connect read dependencies
        for resource in &reads {
            self.connect(pass_id, *resource, ResourceAccess::Read);
        }

        // Connect write dependencies
        for resource in &writes {
            self.connect(pass_id, *resource, ResourceAccess::Write);
        }

        // Handle indirect dispatch buffer (if not already in reads)
        if let super::passes::DispatchSize::Indirect { buffer, .. } = &config.dispatch_size {
            if !reads.contains(buffer) {
                self.connect(pass_id, *buffer, ResourceAccess::Read);
            }
        }

        pass_id
    }

    /// Adds a copy pass to the frame graph with full configuration.
    ///
    /// This method creates a pass node based on the provided copy pass
    /// configuration and automatically sets up resource dependencies based
    /// on the copy operations.
    ///
    /// # Arguments
    ///
    /// * `config` - The copy pass configuration
    ///
    /// # Returns
    ///
    /// The PassId of the newly created pass.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::frame_graph::passes::*;
    /// use renderer_backend::frame_graph::graph::*;
    ///
    /// let mut graph = FrameGraph::new();
    /// let src_buffer = graph.add_resource("src", ResourceType::Buffer, GraphResourceLifetime::Transient);
    /// let dst_buffer = graph.add_resource("dst", ResourceType::Buffer, GraphResourceLifetime::Transient);
    ///
    /// let config = CopyPassBuilder::new("copy_pass")
    ///     .copy_buffer(src_buffer, 0, dst_buffer, 0, 1024)
    ///     .build();
    ///
    /// let pass_id = graph.add_copy_pass(config);
    /// ```
    pub fn add_copy_pass(&mut self, config: super::passes::CopyPassConfig) -> PassId {
        // Create the pass node as a Transfer type
        let pass_id = self.add_pass(&config.name, PassType::Transfer);

        // Connect source resources as reads
        for resource in config.source_resources() {
            self.connect(pass_id, resource, ResourceAccess::Read);
        }

        // Connect destination resources as writes
        for resource in config.destination_resources() {
            self.connect(pass_id, resource, ResourceAccess::Write);
        }

        pass_id
    }

    /// Adds a ray tracing pass to the frame graph with full configuration.
    ///
    /// This method creates a pass node based on the provided ray tracing pass
    /// configuration and automatically sets up resource dependencies based
    /// on the builder's read/write tracking.
    ///
    /// # Arguments
    ///
    /// * `config` - The ray tracing pass configuration
    /// * `reads` - Resources this pass reads from (TLAS, textures, etc.)
    /// * `writes` - Resources this pass writes to (output image, etc.)
    ///
    /// # Returns
    ///
    /// The PassId of the newly created pass.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::frame_graph::passes::*;
    /// use renderer_backend::frame_graph::graph::*;
    ///
    /// let mut graph = FrameGraph::new();
    /// let sbt = graph.add_resource("sbt", ResourceType::Buffer, GraphResourceLifetime::Persistent);
    /// let tlas = graph.add_resource("tlas", ResourceType::AccelerationStructure, GraphResourceLifetime::Persistent);
    /// let output = graph.add_resource("output", ResourceType::Texture2D, GraphResourceLifetime::Transient);
    ///
    /// let (config, reads, writes) = RayTracingPassBuilder::new("primary_rays")
    ///     .shader_binding_table(sbt)
    ///     .dispatch(1920, 1080, 1)
    ///     .read_resource(tlas)
    ///     .write_resource(output)
    ///     .build_with_deps();
    ///
    /// let pass_id = graph.add_ray_tracing_pass(config, &reads, &writes);
    /// ```
    pub fn add_ray_tracing_pass(
        &mut self,
        config: super::passes::RayTracingPassConfig,
        reads: &[ResourceId],
        writes: &[ResourceId],
    ) -> PassId {
        // Create the pass node as a RayTracing type
        let pass_id = self.add_pass(&config.name, PassType::RayTracing);

        // Connect SBT as a read dependency if configured
        if let Some(sbt) = config.shader_binding_table {
            if !reads.contains(&sbt) {
                self.connect(pass_id, sbt, ResourceAccess::Read);
            }
        }

        // Connect indirect dispatch buffer as read dependency if configured
        if let super::passes::RayDispatchSize::Indirect { buffer, .. } = &config.dispatch_size {
            if !reads.contains(buffer) {
                self.connect(pass_id, *buffer, ResourceAccess::Read);
            }
        }

        // Connect read dependencies
        for resource in reads {
            self.connect(pass_id, *resource, ResourceAccess::Read);
        }

        // Connect write dependencies
        for resource in writes {
            self.connect(pass_id, *resource, ResourceAccess::Write);
        }

        pass_id
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for FrameGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameGraph")
            .field("passes", &self.passes.len())
            .field("resources", &self.resources.len())
            .field("compiled", &self.compiled)
            .field("execution_order", &self.execution_order)
            .finish()
    }
}

impl fmt::Display for FrameGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FrameGraph(passes={}, resources={}, compiled={})",
            self.passes.len(),
            self.resources.len(),
            self.compiled
        )
    }
}

// ---------------------------------------------------------------------------
// Frame Graph Builder (Optional Fluent API)
// ---------------------------------------------------------------------------

/// Builder for constructing passes with a fluent API.
pub struct PassBuilder<'a> {
    graph: &'a mut FrameGraph,
    pass_id: PassId,
}

impl<'a> PassBuilder<'a> {
    /// Creates a new pass builder.
    fn new(graph: &'a mut FrameGraph, pass_id: PassId) -> Self {
        Self { graph, pass_id }
    }

    /// Adds an input resource to the pass.
    pub fn read(self, resource: ResourceId) -> Self {
        self.graph.connect(self.pass_id, resource, ResourceAccess::Read);
        self
    }

    /// Adds an output resource to the pass.
    pub fn write(self, resource: ResourceId) -> Self {
        self.graph.connect(self.pass_id, resource, ResourceAccess::Write);
        self
    }

    /// Adds a read-write resource to the pass.
    pub fn read_write(self, resource: ResourceId) -> Self {
        self.graph.connect(self.pass_id, resource, ResourceAccess::ReadWrite);
        self
    }

    /// Sets the pass callback.
    pub fn callback<F>(self, callback: F) -> Self
    where
        F: FnOnce(&mut RenderContext) + Send + Sync + 'static,
    {
        if let Some(pass) = self.graph.passes.get_mut(&self.pass_id) {
            pass.set_callback(callback);
        }
        self
    }

    /// Disables this pass.
    pub fn disable(self) -> Self {
        if let Some(pass) = self.graph.passes.get_mut(&self.pass_id) {
            pass.enabled = false;
        }
        self
    }

    /// Returns the pass ID.
    pub fn id(&self) -> PassId {
        self.pass_id
    }

    /// Finishes building and returns the pass ID.
    pub fn build(self) -> PassId {
        self.pass_id
    }
}

/// Builder for constructing frame graphs with a fluent API.
pub struct FrameGraphBuilder {
    graph: FrameGraph,
}

impl FrameGraphBuilder {
    /// Creates a new frame graph builder.
    pub fn new() -> Self {
        Self {
            graph: FrameGraph::new(),
        }
    }

    /// Adds a resource and returns its ID.
    pub fn add_resource(
        &mut self,
        name: impl Into<String>,
        resource_type: ResourceType,
        lifetime: GraphResourceLifetime,
    ) -> ResourceId {
        self.graph.add_resource(name, resource_type, lifetime)
    }

    /// Adds a pass and returns a builder for it.
    pub fn add_pass(&mut self, name: impl Into<String>, pass_type: PassType) -> PassBuilder<'_> {
        let pass_id = self.graph.add_pass(name, pass_type);
        PassBuilder::new(&mut self.graph, pass_id)
    }

    /// Adds a render pass with full configuration.
    ///
    /// This method creates a render pass based on the provided configuration
    /// and automatically sets up resource dependencies.
    pub fn add_render_pass<E>(&mut self, config: super::passes::RenderPassConfig, executor: E) -> PassId
    where
        E: super::passes::PassExecutor + 'static,
    {
        self.graph.add_render_pass(config, executor)
    }

    /// Adds a compute pass with full configuration.
    ///
    /// This method creates a compute pass based on the provided configuration
    /// and automatically sets up resource dependencies.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::frame_graph::passes::*;
    /// use renderer_backend::frame_graph::graph::*;
    ///
    /// let mut builder = FrameGraphBuilder::new();
    /// let input = builder.add_resource("input", ResourceType::Buffer, GraphResourceLifetime::Transient);
    /// let output = builder.add_resource("output", ResourceType::Buffer, GraphResourceLifetime::Transient);
    ///
    /// let (config, reads, writes) = ComputePassBuilder::new("particle_sim")
    ///     .dispatch(256, 256, 1)
    ///     .read_resource(input)
    ///     .write_resource(output)
    ///     .build_with_deps();
    ///
    /// builder.add_compute_pass(config, reads, writes, NoOpComputeExecutor);
    /// ```
    pub fn add_compute_pass<E>(
        &mut self,
        config: super::passes::ComputePassConfig,
        reads: Vec<ResourceId>,
        writes: Vec<ResourceId>,
        executor: E,
    ) -> PassId
    where
        E: super::passes::ComputePassExecutor + 'static,
    {
        self.graph.add_compute_pass(config, reads, writes, executor)
    }

    /// Adds a copy pass with full configuration.
    ///
    /// This method creates a copy pass based on the provided configuration
    /// and automatically sets up resource dependencies.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use renderer_backend::frame_graph::passes::*;
    /// use renderer_backend::frame_graph::graph::*;
    ///
    /// let mut builder = FrameGraphBuilder::new();
    /// let src = builder.add_resource("src", ResourceType::Buffer, GraphResourceLifetime::Transient);
    /// let dst = builder.add_resource("dst", ResourceType::Buffer, GraphResourceLifetime::Transient);
    ///
    /// let config = CopyPassBuilder::new("buffer_copy")
    ///     .copy_buffer(src, 0, dst, 0, 1024)
    ///     .build();
    ///
    /// builder.add_copy_pass(config);
    /// ```
    pub fn add_copy_pass(&mut self, config: super::passes::CopyPassConfig) -> PassId {
        self.graph.add_copy_pass(config)
    }

    /// Compiles and returns the built frame graph.
    pub fn build(mut self) -> Result<FrameGraph, FrameGraphError> {
        self.graph.compile()?;
        Ok(self.graph)
    }

    /// Returns the frame graph without compiling.
    pub fn build_unchecked(self) -> FrameGraph {
        self.graph
    }
}

impl Default for FrameGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // ResourceId Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_resource_id_new() {
        let id = ResourceId::new(42);
        assert_eq!(id.raw(), 42);
        assert!(!id.is_invalid());
    }

    #[test]
    fn test_resource_id_invalid() {
        let id = ResourceId::INVALID;
        assert!(id.is_invalid());
        assert_eq!(id.raw(), u64::MAX);
    }

    #[test]
    fn test_resource_id_display() {
        let id = ResourceId::new(5);
        assert_eq!(format!("{}", id), "ResourceId(5)");
        assert_eq!(format!("{}", ResourceId::INVALID), "ResourceId::INVALID");
    }

    #[test]
    fn test_resource_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ResourceId::new(1));
        set.insert(ResourceId::new(2));
        assert!(set.contains(&ResourceId::new(1)));
        assert!(!set.contains(&ResourceId::new(3)));
    }

    // ---------------------------------------------------------------------------
    // PassId Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_pass_id_new() {
        let id = PassId::new(100);
        assert_eq!(id.raw(), 100);
        assert!(!id.is_invalid());
    }

    #[test]
    fn test_pass_id_invalid() {
        let id = PassId::INVALID;
        assert!(id.is_invalid());
    }

    #[test]
    fn test_pass_id_equality() {
        let a = PassId::new(10);
        let b = PassId::new(10);
        let c = PassId::new(20);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ---------------------------------------------------------------------------
    // ResourceAccess Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_resource_access_is_read() {
        assert!(ResourceAccess::Read.is_read());
        assert!(!ResourceAccess::Write.is_read());
        assert!(ResourceAccess::ReadWrite.is_read());
    }

    #[test]
    fn test_resource_access_is_write() {
        assert!(!ResourceAccess::Read.is_write());
        assert!(ResourceAccess::Write.is_write());
        assert!(ResourceAccess::ReadWrite.is_write());
    }

    #[test]
    fn test_resource_access_conflicts() {
        // RAW: read after write
        assert!(ResourceAccess::Read.conflicts_with(&ResourceAccess::Write));
        // WAW: write after write
        assert!(ResourceAccess::Write.conflicts_with(&ResourceAccess::Write));
        // WAR: write after read
        assert!(ResourceAccess::Write.conflicts_with(&ResourceAccess::Read));
        // RAR: read after read (no conflict)
        assert!(!ResourceAccess::Read.conflicts_with(&ResourceAccess::Read));
    }

    #[test]
    fn test_resource_access_conflicts_readwrite() {
        // ReadWrite conflicts with everything
        assert!(ResourceAccess::ReadWrite.conflicts_with(&ResourceAccess::Read));
        assert!(ResourceAccess::ReadWrite.conflicts_with(&ResourceAccess::Write));
        assert!(ResourceAccess::ReadWrite.conflicts_with(&ResourceAccess::ReadWrite));
    }

    // ---------------------------------------------------------------------------
    // PassType Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_pass_type_is_graphics() {
        assert!(PassType::Render.is_graphics());
        assert!(!PassType::Compute.is_graphics());
        assert!(!PassType::Transfer.is_graphics());
        assert!(!PassType::RayTracing.is_graphics());
    }

    #[test]
    fn test_pass_type_is_compute() {
        assert!(PassType::Compute.is_compute());
        assert!(!PassType::Render.is_compute());
    }

    #[test]
    fn test_pass_type_display() {
        assert_eq!(format!("{}", PassType::Render), "Render");
        assert_eq!(format!("{}", PassType::Compute), "Compute");
        assert_eq!(format!("{}", PassType::Transfer), "Transfer");
        assert_eq!(format!("{}", PassType::RayTracing), "RayTracing");
    }

    // ---------------------------------------------------------------------------
    // ResourceType Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_resource_type_is_buffer() {
        assert!(ResourceType::Buffer.is_buffer());
        assert!(!ResourceType::Texture2D.is_buffer());
    }

    #[test]
    fn test_resource_type_is_texture() {
        assert!(ResourceType::Texture2D.is_texture());
        assert!(ResourceType::Texture3D.is_texture());
        assert!(ResourceType::TextureCube.is_texture());
        assert!(ResourceType::Texture2DArray.is_texture());
        assert!(!ResourceType::Buffer.is_texture());
    }

    // ---------------------------------------------------------------------------
    // GraphResourceLifetime Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_resource_lifetime_transient() {
        let lt = GraphResourceLifetime::Transient;
        assert!(lt.is_transient());
        assert!(!lt.is_persistent());
        assert!(!lt.is_imported());
        assert!(lt.can_alias());
    }

    #[test]
    fn test_resource_lifetime_persistent() {
        let lt = GraphResourceLifetime::Persistent;
        assert!(!lt.is_transient());
        assert!(lt.is_persistent());
        assert!(!lt.can_alias());
    }

    #[test]
    fn test_resource_lifetime_imported() {
        let lt = GraphResourceLifetime::Imported;
        assert!(lt.is_imported());
        assert!(!lt.can_alias());
    }

    // ---------------------------------------------------------------------------
    // PassNode Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_pass_node_creation() {
        let pass = PassNode::new(PassId::new(0), "test_pass", PassType::Render);
        assert_eq!(pass.id, PassId::new(0));
        assert_eq!(pass.name, "test_pass");
        assert_eq!(pass.pass_type, PassType::Render);
        assert!(pass.inputs.is_empty());
        assert!(pass.outputs.is_empty());
        assert!(pass.enabled);
    }

    #[test]
    fn test_pass_node_add_input_output() {
        let mut pass = PassNode::new(PassId::new(0), "test", PassType::Compute);
        let res_id = ResourceId::new(1);

        pass.add_input_resource(res_id, PipelineStage::ComputeShader);
        pass.add_output_resource(res_id, PipelineStage::ComputeShader);

        assert_eq!(pass.inputs.len(), 1);
        assert_eq!(pass.outputs.len(), 1);
        assert_eq!(pass.inputs[0].resource, res_id);
        assert_eq!(pass.outputs[0].resource, res_id);
    }

    #[test]
    fn test_pass_node_callback() {
        let mut pass = PassNode::new(PassId::new(0), "callback_test", PassType::Compute);
        assert!(!pass.has_callback());

        pass.set_callback(|_ctx| {});
        assert!(pass.has_callback());

        let cb = pass.take_callback();
        assert!(cb.is_some());
        assert!(!pass.has_callback());
    }

    // ---------------------------------------------------------------------------
    // ResourceNode Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_resource_node_creation() {
        let resource = ResourceNode::new(
            ResourceId::new(0),
            "test_texture",
            ResourceType::Texture2D,
            GraphResourceLifetime::Transient,
        );
        assert_eq!(resource.id, ResourceId::new(0));
        assert_eq!(resource.name, "test_texture");
        assert!(resource.is_transient());
        assert!(resource.producer.is_none());
        assert!(resource.consumers.is_empty());
    }

    #[test]
    fn test_resource_node_producer_consumer() {
        let mut resource = ResourceNode::new(
            ResourceId::new(0),
            "buffer",
            ResourceType::Buffer,
            GraphResourceLifetime::Transient,
        );

        resource.set_producer(PassId::new(1));
        resource.add_consumer(PassId::new(2));
        resource.add_consumer(PassId::new(3));

        assert_eq!(resource.producer, Some(PassId::new(1)));
        assert_eq!(resource.consumers.len(), 2);
        assert_eq!(resource.reference_count(), 3);
    }

    #[test]
    fn test_resource_node_no_duplicate_consumers() {
        let mut resource = ResourceNode::new(
            ResourceId::new(0),
            "buffer",
            ResourceType::Buffer,
            GraphResourceLifetime::Transient,
        );

        resource.add_consumer(PassId::new(1));
        resource.add_consumer(PassId::new(1)); // duplicate
        assert_eq!(resource.consumers.len(), 1);
    }

    // ---------------------------------------------------------------------------
    // FrameGraph Basic Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_frame_graph_new() {
        let graph = FrameGraph::new();
        assert_eq!(graph.pass_count(), 0);
        assert_eq!(graph.resource_count(), 0);
        assert!(!graph.is_compiled());
    }

    #[test]
    fn test_frame_graph_add_pass() {
        let mut graph = FrameGraph::new();
        let pass_id = graph.add_pass("render_pass", PassType::Render);
        assert_eq!(graph.pass_count(), 1);
        assert!(graph.get_pass(pass_id).is_some());
        assert_eq!(graph.get_pass(pass_id).unwrap().name, "render_pass");
    }

    #[test]
    fn test_frame_graph_add_resource() {
        let mut graph = FrameGraph::new();
        let res_id = graph.add_resource("color_buffer", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        assert_eq!(graph.resource_count(), 1);
        assert!(graph.get_resource(res_id).is_some());
    }

    #[test]
    fn test_frame_graph_connect() {
        let mut graph = FrameGraph::new();
        let pass = graph.add_pass("test_pass", PassType::Render);
        let res = graph.add_resource("texture", ResourceType::Texture2D, GraphResourceLifetime::Transient);

        graph.connect(pass, res, ResourceAccess::Write);

        let pass_node = graph.get_pass(pass).unwrap();
        assert_eq!(pass_node.outputs.len(), 1);

        let res_node = graph.get_resource(res).unwrap();
        assert_eq!(res_node.producer, Some(pass));
    }

    // ---------------------------------------------------------------------------
    // FrameGraph Compilation Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_frame_graph_compile_empty() {
        let mut graph = FrameGraph::new();
        assert!(graph.compile().is_ok());
        assert!(graph.is_compiled());
    }

    #[test]
    fn test_frame_graph_compile_single_pass() {
        let mut graph = FrameGraph::new();
        let pass = graph.add_pass("single", PassType::Render);
        let res = graph.add_resource("output", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        graph.connect(pass, res, ResourceAccess::Write);

        assert!(graph.compile().is_ok());
        assert!(graph.execution_order().contains(&pass));
    }

    #[test]
    fn test_frame_graph_compile_linear_chain() {
        let mut graph = FrameGraph::new();

        // A -> B -> C (linear dependency chain)
        let pass_a = graph.add_pass("pass_a", PassType::Render);
        let pass_b = graph.add_pass("pass_b", PassType::Compute);
        let pass_c = graph.add_pass("pass_c", PassType::Render);

        let res_ab = graph.add_resource("res_ab", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        let res_bc = graph.add_resource("res_bc", ResourceType::Buffer, GraphResourceLifetime::Transient);

        graph.connect(pass_a, res_ab, ResourceAccess::Write);
        graph.connect(pass_b, res_ab, ResourceAccess::Read);
        graph.connect(pass_b, res_bc, ResourceAccess::Write);
        graph.connect(pass_c, res_bc, ResourceAccess::Read);

        assert!(graph.compile().is_ok());

        // Verify order: A before B before C
        let order = graph.execution_order();
        let pos_a = order.iter().position(|&id| id == pass_a).unwrap();
        let pos_b = order.iter().position(|&id| id == pass_b).unwrap();
        let pos_c = order.iter().position(|&id| id == pass_c).unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_frame_graph_compile_missing_resource() {
        let mut graph = FrameGraph::new();
        let pass = graph.add_pass("test", PassType::Render);

        // Manually add a reference to a non-existent resource
        if let Some(p) = graph.get_pass_mut(pass) {
            p.add_input(ResourceUsage::read(ResourceId::new(999)));
        }

        let result = graph.compile();
        assert!(matches!(result, Err(FrameGraphError::MissingResource(_))));
    }

    #[test]
    fn test_frame_graph_compile_cycle_detection() {
        let mut graph = FrameGraph::new();

        // Create a cycle: A -> B -> A
        let pass_a = graph.add_pass("pass_a", PassType::Render);
        let pass_b = graph.add_pass("pass_b", PassType::Render);

        let res_ab = graph.add_resource("res_ab", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        let res_ba = graph.add_resource("res_ba", ResourceType::Texture2D, GraphResourceLifetime::Transient);

        // A writes res_ab, B reads res_ab
        graph.connect(pass_a, res_ab, ResourceAccess::Write);
        graph.connect(pass_b, res_ab, ResourceAccess::Read);

        // B writes res_ba, A reads res_ba (creates cycle)
        graph.connect(pass_b, res_ba, ResourceAccess::Write);
        graph.connect(pass_a, res_ba, ResourceAccess::Read);

        let result = graph.compile();
        assert!(matches!(result, Err(FrameGraphError::CyclicDependency)));
    }

    // ---------------------------------------------------------------------------
    // FrameGraph Execution Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_frame_graph_execute_not_compiled() {
        let mut graph = FrameGraph::new();
        graph.add_pass("test", PassType::Render);

        let mut ctx = RenderContext::new(0);
        let result = graph.execute(&mut ctx);
        assert!(matches!(result, Err(FrameGraphError::NotCompiled)));
    }

    #[test]
    fn test_frame_graph_execute_with_callbacks() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let mut graph = FrameGraph::new();
        let counter = Arc::new(AtomicU32::new(0));

        let counter_a = Arc::clone(&counter);
        let pass_a = graph.add_pass("pass_a", PassType::Render);
        if let Some(p) = graph.get_pass_mut(pass_a) {
            p.set_callback(move |_ctx| {
                counter_a.fetch_add(1, Ordering::SeqCst);
            });
        }

        let counter_b = Arc::clone(&counter);
        let pass_b = graph.add_pass("pass_b", PassType::Render);
        if let Some(p) = graph.get_pass_mut(pass_b) {
            p.set_callback(move |_ctx| {
                counter_b.fetch_add(10, Ordering::SeqCst);
            });
        }

        let res = graph.add_resource("res", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        graph.connect(pass_a, res, ResourceAccess::Write);
        graph.connect(pass_b, res, ResourceAccess::Read);

        graph.compile().unwrap();

        let mut ctx = RenderContext::new(0);
        graph.execute(&mut ctx).unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 11);
    }

    #[test]
    fn test_frame_graph_execute_disabled_pass() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let mut graph = FrameGraph::new();
        let executed = Arc::new(AtomicBool::new(false));

        let pass = graph.add_pass("disabled_pass", PassType::Render);
        let executed_clone = Arc::clone(&executed);
        if let Some(p) = graph.get_pass_mut(pass) {
            p.enabled = false;
            p.set_callback(move |_ctx| {
                executed_clone.store(true, Ordering::SeqCst);
            });
        }

        graph.compile().unwrap();

        let mut ctx = RenderContext::new(0);
        graph.execute(&mut ctx).unwrap();

        // Disabled pass should not execute
        assert!(!executed.load(Ordering::SeqCst));
    }

    // ---------------------------------------------------------------------------
    // FrameGraph Reset and Clear Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_frame_graph_reset() {
        let mut graph = FrameGraph::new();
        let pass = graph.add_pass("test", PassType::Render);
        let res = graph.add_resource("res", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        graph.connect(pass, res, ResourceAccess::Write);
        graph.compile().unwrap();

        graph.reset();

        assert!(!graph.is_compiled());
        // Passes and resources still exist
        assert_eq!(graph.pass_count(), 1);
        assert_eq!(graph.resource_count(), 1);
        // But connections are cleared
        assert!(graph.get_pass(pass).unwrap().inputs.is_empty());
        assert!(graph.get_pass(pass).unwrap().outputs.is_empty());
    }

    #[test]
    fn test_frame_graph_clear() {
        let mut graph = FrameGraph::new();
        graph.add_pass("test", PassType::Render);
        graph.add_resource("res", ResourceType::Texture2D, GraphResourceLifetime::Transient);

        graph.clear();

        assert_eq!(graph.pass_count(), 0);
        assert_eq!(graph.resource_count(), 0);
    }

    // ---------------------------------------------------------------------------
    // FrameGraphBuilder Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_frame_graph_builder_basic() {
        let mut builder = FrameGraphBuilder::new();

        let color = builder.add_resource("color", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        let depth = builder.add_resource("depth", ResourceType::Texture2D, GraphResourceLifetime::Transient);

        builder.add_pass("shadow", PassType::Render).write(depth).build();
        builder.add_pass("main", PassType::Render).read(depth).write(color).build();

        let graph = builder.build().unwrap();

        assert_eq!(graph.pass_count(), 2);
        assert_eq!(graph.resource_count(), 2);
        assert!(graph.is_compiled());
    }

    #[test]
    fn test_frame_graph_builder_with_callback() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let mut builder = FrameGraphBuilder::new();
        let executed = Arc::new(AtomicBool::new(false));

        let res = builder.add_resource("buffer", ResourceType::Buffer, GraphResourceLifetime::Transient);

        let executed_clone = Arc::clone(&executed);
        builder
            .add_pass("compute", PassType::Compute)
            .write(res)
            .callback(move |_ctx| {
                executed_clone.store(true, Ordering::SeqCst);
            })
            .build();

        let mut graph = builder.build().unwrap();
        let mut ctx = RenderContext::new(0);
        graph.execute(&mut ctx).unwrap();

        assert!(executed.load(Ordering::SeqCst));
    }

    // ---------------------------------------------------------------------------
    // FrameGraphError Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_frame_graph_error_display() {
        assert_eq!(
            format!("{}", FrameGraphError::CyclicDependency),
            "Cyclic dependency detected in frame graph"
        );
        assert_eq!(
            format!("{}", FrameGraphError::MissingResource(ResourceId::new(5))),
            "Missing resource: ResourceId(5)"
        );
        assert_eq!(
            format!("{}", FrameGraphError::MissingPass(PassId::new(10))),
            "Missing pass: PassId(10)"
        );
    }

    // ---------------------------------------------------------------------------
    // ResourceUsage Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_resource_usage_helpers() {
        let res_id = ResourceId::new(1);

        let read_usage = ResourceUsage::read(res_id);
        assert!(read_usage.access.is_read());
        assert!(!read_usage.access.is_write());

        let write_usage = ResourceUsage::write(res_id);
        assert!(write_usage.access.is_write());

        let rw_usage = ResourceUsage::read_write(res_id);
        assert!(rw_usage.access.is_read());
        assert!(rw_usage.access.is_write());
    }

    // ---------------------------------------------------------------------------
    // Complex Scenario Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_frame_graph_diamond_dependency() {
        // Diamond pattern: A -> B, A -> C, B -> D, C -> D
        let mut graph = FrameGraph::new();

        let pass_a = graph.add_pass("pass_a", PassType::Render);
        let pass_b = graph.add_pass("pass_b", PassType::Compute);
        let pass_c = graph.add_pass("pass_c", PassType::Compute);
        let pass_d = graph.add_pass("pass_d", PassType::Render);

        let res_ab = graph.add_resource("res_ab", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        let res_ac = graph.add_resource("res_ac", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        let res_bd = graph.add_resource("res_bd", ResourceType::Buffer, GraphResourceLifetime::Transient);
        let res_cd = graph.add_resource("res_cd", ResourceType::Buffer, GraphResourceLifetime::Transient);

        graph.connect(pass_a, res_ab, ResourceAccess::Write);
        graph.connect(pass_b, res_ab, ResourceAccess::Read);
        graph.connect(pass_a, res_ac, ResourceAccess::Write);
        graph.connect(pass_c, res_ac, ResourceAccess::Read);
        graph.connect(pass_b, res_bd, ResourceAccess::Write);
        graph.connect(pass_d, res_bd, ResourceAccess::Read);
        graph.connect(pass_c, res_cd, ResourceAccess::Write);
        graph.connect(pass_d, res_cd, ResourceAccess::Read);

        assert!(graph.compile().is_ok());

        let order = graph.execution_order();
        let pos_a = order.iter().position(|&id| id == pass_a).unwrap();
        let pos_b = order.iter().position(|&id| id == pass_b).unwrap();
        let pos_c = order.iter().position(|&id| id == pass_c).unwrap();
        let pos_d = order.iter().position(|&id| id == pass_d).unwrap();

        // A must come before B and C
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        // B and C must come before D
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_frame_graph_find_readers_writers() {
        let mut graph = FrameGraph::new();

        let pass_a = graph.add_pass("writer", PassType::Render);
        let pass_b = graph.add_pass("reader1", PassType::Render);
        let pass_c = graph.add_pass("reader2", PassType::Render);

        let res = graph.add_resource("shared", ResourceType::Texture2D, GraphResourceLifetime::Transient);

        graph.connect(pass_a, res, ResourceAccess::Write);
        graph.connect(pass_b, res, ResourceAccess::Read);
        graph.connect(pass_c, res, ResourceAccess::Read);

        let writers = graph.find_writers(res);
        let readers = graph.find_readers(res);

        assert_eq!(writers.len(), 1);
        assert!(writers.contains(&pass_a));
        assert_eq!(readers.len(), 2);
        assert!(readers.contains(&pass_b));
        assert!(readers.contains(&pass_c));
    }

    #[test]
    fn test_frame_graph_multiple_resources() {
        let mut graph = FrameGraph::new();

        // Create multiple resources and passes
        let gbuffer_albedo = graph.add_resource("gbuffer_albedo", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        let gbuffer_normal = graph.add_resource("gbuffer_normal", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        let gbuffer_depth = graph.add_resource("gbuffer_depth", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        let lighting_output = graph.add_resource("lighting", ResourceType::Texture2D, GraphResourceLifetime::Transient);
        let final_output = graph.add_resource("final", ResourceType::Texture2D, GraphResourceLifetime::Imported);

        let gbuffer_pass = graph.add_pass("gbuffer", PassType::Render);
        let lighting_pass = graph.add_pass("lighting", PassType::Compute);
        let composite_pass = graph.add_pass("composite", PassType::Render);

        // GBuffer writes all gbuffer targets
        graph.connect(gbuffer_pass, gbuffer_albedo, ResourceAccess::Write);
        graph.connect(gbuffer_pass, gbuffer_normal, ResourceAccess::Write);
        graph.connect(gbuffer_pass, gbuffer_depth, ResourceAccess::Write);

        // Lighting reads gbuffer, writes lighting output
        graph.connect(lighting_pass, gbuffer_albedo, ResourceAccess::Read);
        graph.connect(lighting_pass, gbuffer_normal, ResourceAccess::Read);
        graph.connect(lighting_pass, gbuffer_depth, ResourceAccess::Read);
        graph.connect(lighting_pass, lighting_output, ResourceAccess::Write);

        // Composite reads lighting, writes final
        graph.connect(composite_pass, lighting_output, ResourceAccess::Read);
        graph.connect(composite_pass, final_output, ResourceAccess::Write);

        assert!(graph.compile().is_ok());

        // Verify execution order
        let order = graph.execution_order();
        let pos_gbuffer = order.iter().position(|&id| id == gbuffer_pass).unwrap();
        let pos_lighting = order.iter().position(|&id| id == lighting_pass).unwrap();
        let pos_composite = order.iter().position(|&id| id == composite_pass).unwrap();

        assert!(pos_gbuffer < pos_lighting);
        assert!(pos_lighting < pos_composite);
    }
}
