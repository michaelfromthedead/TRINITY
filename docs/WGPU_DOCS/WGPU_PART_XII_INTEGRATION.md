# WGPU_PART_XII_INTEGRATION.md — TRINITY Integration

> **Scope**: Frame graph integration, Python bridge, resource management, and complete TRINITY-wgpu architecture
> **Purpose**: Connect wgpu primitives to TRINITY's high-level rendering abstractions
> **wgpu Version**: 25.x+

---

# Chapter 23: Frame Graph Integration

The frame graph is TRINITY's central abstraction for organizing GPU work. It provides automatic resource lifetime management, barrier placement, and execution scheduling.

---

## 23.1 Resource Declaration

### 23.1.1 Virtual Resources

Virtual resources are declared at frame graph construction time but allocated lazily during execution.

```rust
pub struct FrameGraph {
    passes: Vec<PassNode>,
    resources: HashMap<ResourceId, ResourceNode>,
    edges: Vec<ResourceEdge>,
    execution_order: Vec<usize>,
    compiled: bool,
}

pub struct ResourceNode {
    id: ResourceId,
    name: String,
    descriptor: ResourceDescriptor,
    lifetime: ResourceLifetime,
    physical: Option<PhysicalResource>,
}

pub enum ResourceDescriptor {
    Buffer(BufferResourceDesc),
    Texture(TextureResourceDesc),
}

pub struct BufferResourceDesc {
    pub size: u64,
    pub usage: wgpu::BufferUsages,
}

pub struct TextureResourceDesc {
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    pub mip_count: u32,
    pub sample_count: u32,
}

impl FrameGraph {
    pub fn create_virtual_texture(
        &mut self,
        name: &str,
        desc: TextureResourceDesc,
    ) -> ResourceId {
        let id = ResourceId::new();
        
        self.resources.insert(id, ResourceNode {
            id,
            name: name.to_string(),
            descriptor: ResourceDescriptor::Texture(desc),
            lifetime: ResourceLifetime::Transient,
            physical: None,
        });
        
        id
    }
    
    pub fn create_virtual_buffer(
        &mut self,
        name: &str,
        desc: BufferResourceDesc,
    ) -> ResourceId {
        let id = ResourceId::new();
        
        self.resources.insert(id, ResourceNode {
            id,
            name: name.to_string(),
            descriptor: ResourceDescriptor::Buffer(desc),
            lifetime: ResourceLifetime::Transient,
            physical: None,
        });
        
        id
    }
}
```

### 23.1.2 Transient Resources

Transient resources exist only for the duration of a frame and can be aliased:

```rust
pub enum ResourceLifetime {
    Transient,          // Lives only within frame, can be aliased
    Persistent,         // Lives across frames
    External,           // Owned outside frame graph
    Imported,           // Imported from previous frame
}

pub struct TransientResourcePool {
    texture_pool: Vec<PooledTexture>,
    buffer_pool: Vec<PooledBuffer>,
    frame_allocations: Vec<AllocationRecord>,
}

pub struct PooledTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    desc: TextureResourceDesc,
    last_used_frame: u64,
    in_use: bool,
}

impl TransientResourcePool {
    pub fn acquire_texture(
        &mut self,
        device: &wgpu::Device,
        desc: &TextureResourceDesc,
        frame: u64,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        // Try to find compatible pooled texture
        for pooled in &mut self.texture_pool {
            if !pooled.in_use && Self::is_compatible(&pooled.desc, desc) {
                pooled.in_use = true;
                pooled.last_used_frame = frame;
                return (pooled.texture.clone(), pooled.view.clone());
            }
        }
        
        // Create new texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TransientTexture"),
            size: desc.size,
            mip_level_count: desc.mip_count,
            sample_count: desc.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        });
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.texture_pool.push(PooledTexture {
            texture: texture.clone(),
            view: view.clone(),
            desc: desc.clone(),
            last_used_frame: frame,
            in_use: true,
        });
        
        (texture, view)
    }
    
    fn is_compatible(pooled: &TextureResourceDesc, requested: &TextureResourceDesc) -> bool {
        pooled.size.width >= requested.size.width
            && pooled.size.height >= requested.size.height
            && pooled.format == requested.format
            && pooled.sample_count == requested.sample_count
            && pooled.usage.contains(requested.usage)
    }
    
    pub fn release_all(&mut self) {
        for pooled in &mut self.texture_pool {
            pooled.in_use = false;
        }
        for pooled in &mut self.buffer_pool {
            pooled.in_use = false;
        }
    }
    
    pub fn gc(&mut self, current_frame: u64, max_unused_frames: u64) {
        self.texture_pool.retain(|p| {
            current_frame - p.last_used_frame < max_unused_frames
        });
        self.buffer_pool.retain(|p| {
            current_frame - p.last_used_frame < max_unused_frames
        });
    }
}
```

### 23.1.3 External Resources

External resources are provided from outside the frame graph:

```rust
impl FrameGraph {
    pub fn import_texture(
        &mut self,
        name: &str,
        texture: &wgpu::Texture,
        view: &wgpu::TextureView,
        desc: TextureResourceDesc,
    ) -> ResourceId {
        let id = ResourceId::new();
        
        self.resources.insert(id, ResourceNode {
            id,
            name: name.to_string(),
            descriptor: ResourceDescriptor::Texture(desc),
            lifetime: ResourceLifetime::External,
            physical: Some(PhysicalResource::Texture {
                texture: texture.clone(),
                view: view.clone(),
            }),
        });
        
        id
    }
    
    pub fn import_swapchain(
        &mut self,
        surface_texture: &wgpu::SurfaceTexture,
        view: &wgpu::TextureView,
    ) -> ResourceId {
        let id = ResourceId::new();
        let size = surface_texture.texture.size();
        
        self.resources.insert(id, ResourceNode {
            id,
            name: "Swapchain".to_string(),
            descriptor: ResourceDescriptor::Texture(TextureResourceDesc {
                size,
                format: surface_texture.texture.format(),
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                mip_count: 1,
                sample_count: 1,
            }),
            lifetime: ResourceLifetime::External,
            physical: Some(PhysicalResource::SwapchainTexture {
                view: view.clone(),
            }),
        });
        
        id
    }
}
```

### 23.1.4 Resource Aliasing

Aliasing allows multiple transient resources to share the same physical memory:

```rust
pub struct AliasingInfo {
    pub resource: ResourceId,
    pub first_use: usize,   // First pass index
    pub last_use: usize,    // Last pass index
}

pub struct AliasingAnalyzer;

impl AliasingAnalyzer {
    pub fn compute_aliasing(
        resources: &[ResourceNode],
        passes: &[PassNode],
    ) -> Vec<AliasingGroup> {
        let mut lifetimes = Vec::new();
        
        // Compute lifetime for each transient resource
        for resource in resources {
            if resource.lifetime != ResourceLifetime::Transient {
                continue;
            }
            
            let mut first_use = usize::MAX;
            let mut last_use = 0;
            
            for (pass_idx, pass) in passes.iter().enumerate() {
                if pass.reads.contains(&resource.id) || pass.writes.contains(&resource.id) {
                    first_use = first_use.min(pass_idx);
                    last_use = last_use.max(pass_idx);
                }
            }
            
            lifetimes.push(AliasingInfo {
                resource: resource.id,
                first_use,
                last_use,
            });
        }
        
        // Group non-overlapping resources
        let mut groups: Vec<AliasingGroup> = Vec::new();
        
        for lifetime in lifetimes {
            let mut placed = false;
            
            for group in &mut groups {
                if !group.overlaps(&lifetime) {
                    group.add(lifetime.clone());
                    placed = true;
                    break;
                }
            }
            
            if !placed {
                let mut group = AliasingGroup::new();
                group.add(lifetime);
                groups.push(group);
            }
        }
        
        groups
    }
}

pub struct AliasingGroup {
    resources: Vec<AliasingInfo>,
}

impl AliasingGroup {
    fn overlaps(&self, info: &AliasingInfo) -> bool {
        for existing in &self.resources {
            if !(info.last_use < existing.first_use || info.first_use > existing.last_use) {
                return true;
            }
        }
        false
    }
}
```

---

## 23.2 Pass Declaration

### 23.2.1 Render Passes

```rust
pub struct PassNode {
    id: PassId,
    name: String,
    pass_type: PassType,
    reads: Vec<ResourceId>,
    writes: Vec<ResourceId>,
    execute: Box<dyn PassExecutor>,
}

pub enum PassType {
    Render(RenderPassConfig),
    Compute(ComputePassConfig),
    RayTracing(RTPassConfig),
    Copy,
}

pub struct RenderPassConfig {
    pub color_attachments: Vec<ColorAttachmentConfig>,
    pub depth_attachment: Option<DepthAttachmentConfig>,
    pub resolve_attachments: Vec<ResolveConfig>,
}

pub struct ColorAttachmentConfig {
    pub resource: ResourceId,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_value: wgpu::Color,
}

pub enum LoadOp {
    Load,
    Clear,
    DontCare,
}

pub enum StoreOp {
    Store,
    Discard,
}

impl FrameGraph {
    pub fn add_render_pass<F>(
        &mut self,
        name: &str,
        config: RenderPassConfig,
        execute: F,
    ) -> PassId
    where
        F: FnMut(&mut RenderPassContext) + 'static,
    {
        let id = PassId::new();
        
        // Extract resource dependencies
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        
        for color in &config.color_attachments {
            writes.push(color.resource);
            if matches!(color.load_op, LoadOp::Load) {
                reads.push(color.resource);
            }
        }
        
        if let Some(ref depth) = config.depth_attachment {
            writes.push(depth.resource);
            if matches!(depth.load_op, LoadOp::Load) {
                reads.push(depth.resource);
            }
        }
        
        self.passes.push(PassNode {
            id,
            name: name.to_string(),
            pass_type: PassType::Render(config),
            reads,
            writes,
            execute: Box::new(FnPassExecutor::new(execute)),
        });
        
        id
    }
}

pub struct RenderPassContext<'a> {
    pub pass: wgpu::RenderPass<'a>,
    pub resources: &'a ResolvedResources,
}
```

### 23.2.2 Compute Passes

```rust
pub struct ComputePassConfig {
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
}

impl FrameGraph {
    pub fn add_compute_pass<F>(
        &mut self,
        name: &str,
        config: ComputePassConfig,
        execute: F,
    ) -> PassId
    where
        F: FnMut(&mut ComputePassContext) + 'static,
    {
        let id = PassId::new();
        
        self.passes.push(PassNode {
            id,
            name: name.to_string(),
            pass_type: PassType::Compute(config.clone()),
            reads: config.reads,
            writes: config.writes,
            execute: Box::new(FnPassExecutor::new(execute)),
        });
        
        id
    }
}

pub struct ComputePassContext<'a> {
    pub pass: wgpu::ComputePass<'a>,
    pub resources: &'a ResolvedResources,
}
```

### 23.2.3 Ray Tracing Passes

```rust
pub struct RTPassConfig {
    pub acceleration_structure: ResourceId,
    pub output_image: ResourceId,
    pub shader_binding_table: ResourceId,
}

impl FrameGraph {
    pub fn add_rt_pass<F>(
        &mut self,
        name: &str,
        config: RTPassConfig,
        execute: F,
    ) -> PassId
    where
        F: FnMut(&mut RTPassContext) + 'static,
    {
        let id = PassId::new();
        
        self.passes.push(PassNode {
            id,
            name: name.to_string(),
            pass_type: PassType::RayTracing(config.clone()),
            reads: vec![config.acceleration_structure, config.shader_binding_table],
            writes: vec![config.output_image],
            execute: Box::new(FnPassExecutor::new(execute)),
        });
        
        id
    }
}
```

### 23.2.4 Copy Passes

```rust
impl FrameGraph {
    pub fn add_copy_pass(
        &mut self,
        name: &str,
        src: ResourceId,
        dst: ResourceId,
    ) -> PassId {
        let id = PassId::new();
        
        self.passes.push(PassNode {
            id,
            name: name.to_string(),
            pass_type: PassType::Copy,
            reads: vec![src],
            writes: vec![dst],
            execute: Box::new(CopyPassExecutor { src, dst }),
        });
        
        id
    }
}

struct CopyPassExecutor {
    src: ResourceId,
    dst: ResourceId,
}

impl PassExecutor for CopyPassExecutor {
    fn execute(&mut self, encoder: &mut wgpu::CommandEncoder, resources: &ResolvedResources) {
        match (resources.get(self.src), resources.get(self.dst)) {
            (PhysicalResource::Buffer { buffer: src, .. },
             PhysicalResource::Buffer { buffer: dst, .. }) => {
                encoder.copy_buffer_to_buffer(src, 0, dst, 0, src.size());
            }
            (PhysicalResource::Texture { texture: src, .. },
             PhysicalResource::Texture { texture: dst, .. }) => {
                let size = src.size();
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: src,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: dst,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    size,
                );
            }
            _ => panic!("Invalid copy: mismatched resource types"),
        }
    }
}
```

---

## 23.3 Barrier Resolution

### 23.3.1 Automatic Barrier Placement

```rust
pub struct BarrierResolver {
    resource_states: HashMap<ResourceId, ResourceState>,
}

pub struct ResourceState {
    pub stage: PipelineStage,
    pub access: AccessFlags,
    pub layout: TextureLayout,
}

impl BarrierResolver {
    pub fn new() -> Self {
        Self {
            resource_states: HashMap::new(),
        }
    }
    
    pub fn compute_barriers(
        &mut self,
        passes: &[PassNode],
    ) -> Vec<PassBarriers> {
        let mut all_barriers = Vec::new();
        
        for (pass_idx, pass) in passes.iter().enumerate() {
            let mut barriers = PassBarriers::new(pass_idx);
            
            // Compute read barriers
            for &resource in &pass.reads {
                let current = self.resource_states.get(&resource)
                    .cloned()
                    .unwrap_or(ResourceState::undefined());
                
                let required = self.required_state_for_read(&pass.pass_type, resource);
                
                if current.needs_barrier_to(&required) {
                    barriers.add(Barrier {
                        resource,
                        from: current.clone(),
                        to: required.clone(),
                    });
                }
                
                self.resource_states.insert(resource, required);
            }
            
            // Compute write barriers
            for &resource in &pass.writes {
                let current = self.resource_states.get(&resource)
                    .cloned()
                    .unwrap_or(ResourceState::undefined());
                
                let required = self.required_state_for_write(&pass.pass_type, resource);
                
                if current.needs_barrier_to(&required) {
                    barriers.add(Barrier {
                        resource,
                        from: current.clone(),
                        to: required.clone(),
                    });
                }
                
                self.resource_states.insert(resource, required);
            }
            
            all_barriers.push(barriers);
        }
        
        all_barriers
    }
    
    fn required_state_for_read(&self, pass_type: &PassType, resource: ResourceId) -> ResourceState {
        match pass_type {
            PassType::Render(_) => ResourceState {
                stage: PipelineStage::FRAGMENT,
                access: AccessFlags::SHADER_READ,
                layout: TextureLayout::ShaderReadOnly,
            },
            PassType::Compute(_) => ResourceState {
                stage: PipelineStage::COMPUTE,
                access: AccessFlags::SHADER_READ,
                layout: TextureLayout::General,
            },
            PassType::RayTracing(_) => ResourceState {
                stage: PipelineStage::RAY_TRACING,
                access: AccessFlags::SHADER_READ,
                layout: TextureLayout::General,
            },
            PassType::Copy => ResourceState {
                stage: PipelineStage::COPY,
                access: AccessFlags::COPY_READ,
                layout: TextureLayout::CopySrc,
            },
        }
    }
    
    fn required_state_for_write(&self, pass_type: &PassType, resource: ResourceId) -> ResourceState {
        match pass_type {
            PassType::Render(_) => ResourceState {
                stage: PipelineStage::COLOR_OUTPUT,
                access: AccessFlags::COLOR_WRITE,
                layout: TextureLayout::ColorAttachment,
            },
            PassType::Compute(_) => ResourceState {
                stage: PipelineStage::COMPUTE,
                access: AccessFlags::SHADER_WRITE,
                layout: TextureLayout::General,
            },
            PassType::RayTracing(_) => ResourceState {
                stage: PipelineStage::RAY_TRACING,
                access: AccessFlags::SHADER_WRITE,
                layout: TextureLayout::General,
            },
            PassType::Copy => ResourceState {
                stage: PipelineStage::COPY,
                access: AccessFlags::COPY_WRITE,
                layout: TextureLayout::CopyDst,
            },
        }
    }
}
```

### 23.3.2 Resource State Tracking

```rust
impl ResourceState {
    pub fn undefined() -> Self {
        Self {
            stage: PipelineStage::TOP,
            access: AccessFlags::empty(),
            layout: TextureLayout::Undefined,
        }
    }
    
    pub fn needs_barrier_to(&self, other: &Self) -> bool {
        // Write-after-read
        if self.access.intersects(AccessFlags::READ) && other.access.intersects(AccessFlags::WRITE) {
            return true;
        }
        // Read-after-write
        if self.access.intersects(AccessFlags::WRITE) && other.access.intersects(AccessFlags::READ) {
            return true;
        }
        // Write-after-write
        if self.access.intersects(AccessFlags::WRITE) && other.access.intersects(AccessFlags::WRITE) {
            return true;
        }
        // Layout transition
        if self.layout != other.layout {
            return true;
        }
        false
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TextureLayout {
    Undefined,
    General,
    ColorAttachment,
    DepthAttachment,
    ShaderReadOnly,
    CopySrc,
    CopyDst,
    Present,
}
```

### 23.3.3 Barrier Batching

```rust
impl BarrierResolver {
    pub fn batch_barriers(barriers: Vec<PassBarriers>) -> Vec<BatchedBarriers> {
        let mut batched = Vec::new();
        
        for pass_barriers in barriers {
            if pass_barriers.barriers.is_empty() {
                continue;
            }
            
            // Group barriers by stage
            let mut by_stage: HashMap<(PipelineStage, PipelineStage), Vec<Barrier>> = HashMap::new();
            
            for barrier in pass_barriers.barriers {
                let key = (barrier.from.stage, barrier.to.stage);
                by_stage.entry(key).or_default().push(barrier);
            }
            
            for ((from_stage, to_stage), barriers) in by_stage {
                batched.push(BatchedBarriers {
                    pass_index: pass_barriers.pass_index,
                    src_stage: from_stage,
                    dst_stage: to_stage,
                    barriers,
                });
            }
        }
        
        batched
    }
}

pub struct BatchedBarriers {
    pass_index: usize,
    src_stage: PipelineStage,
    dst_stage: PipelineStage,
    barriers: Vec<Barrier>,
}
```

### 23.3.4 Aliasing Barriers

```rust
pub struct AliasingBarrier {
    pub before: ResourceId,
    pub after: ResourceId,
    pub physical_memory: PhysicalMemoryId,
}

impl BarrierResolver {
    pub fn compute_aliasing_barriers(
        aliasing_groups: &[AliasingGroup],
        passes: &[PassNode],
    ) -> Vec<AliasingBarrier> {
        let mut barriers = Vec::new();
        
        for group in aliasing_groups {
            let resources: Vec<_> = group.resources.iter()
                .map(|info| (info.resource, info.first_use, info.last_use))
                .collect();
            
            // Sort by first use
            let mut sorted = resources.clone();
            sorted.sort_by_key(|(_, first, _)| *first);
            
            // Create aliasing barriers between consecutive uses
            for window in sorted.windows(2) {
                let (prev_resource, _, prev_last) = window[0];
                let (next_resource, next_first, _) = window[1];
                
                // If they don't overlap, we need an aliasing barrier
                if prev_last < next_first {
                    barriers.push(AliasingBarrier {
                        before: prev_resource,
                        after: next_resource,
                        physical_memory: PhysicalMemoryId::new(),
                    });
                }
            }
        }
        
        barriers
    }
}
```

---

## 23.4 Execution

### 23.4.1 Pass Scheduling

```rust
impl FrameGraph {
    pub fn compile(&mut self) {
        // Build dependency graph
        let dependencies = self.build_dependency_graph();
        
        // Topological sort
        self.execution_order = self.topological_sort(&dependencies);
        
        // Compute aliasing
        let aliasing = AliasingAnalyzer::compute_aliasing(
            &self.resources.values().collect::<Vec<_>>(),
            &self.passes,
        );
        
        // Resolve barriers
        let mut resolver = BarrierResolver::new();
        self.barriers = resolver.compute_barriers(&self.passes);
        
        self.compiled = true;
    }
    
    fn build_dependency_graph(&self) -> HashMap<usize, Vec<usize>> {
        let mut deps: HashMap<usize, Vec<usize>> = HashMap::new();
        
        // Resource to last writer
        let mut last_writer: HashMap<ResourceId, usize> = HashMap::new();
        
        for (pass_idx, pass) in self.passes.iter().enumerate() {
            let mut pass_deps = Vec::new();
            
            // Depend on writers of our reads
            for &read in &pass.reads {
                if let Some(&writer) = last_writer.get(&read) {
                    pass_deps.push(writer);
                }
            }
            
            deps.insert(pass_idx, pass_deps);
            
            // Update last writer for our writes
            for &write in &pass.writes {
                last_writer.insert(write, pass_idx);
            }
        }
        
        deps
    }
    
    fn topological_sort(&self, deps: &HashMap<usize, Vec<usize>>) -> Vec<usize> {
        let n = self.passes.len();
        let mut in_degree = vec![0; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        
        for (&node, dependencies) in deps {
            in_degree[node] = dependencies.len();
            for &dep in dependencies {
                adj[dep].push(node);
            }
        }
        
        let mut queue: VecDeque<usize> = in_degree.iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(i, _)| i)
            .collect();
        
        let mut order = Vec::with_capacity(n);
        
        while let Some(node) = queue.pop_front() {
            order.push(node);
            
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }
        
        assert_eq!(order.len(), n, "Cycle detected in frame graph");
        order
    }
}
```

### 23.4.2 Async Compute Overlap

```rust
pub struct AsyncComputeScheduler {
    graphics_timeline: Vec<PassId>,
    compute_timeline: Vec<PassId>,
    sync_points: Vec<SyncPoint>,
}

pub struct SyncPoint {
    pub graphics_pass: PassId,
    pub compute_pass: PassId,
    pub direction: SyncDirection,
}

pub enum SyncDirection {
    GraphicsToCompute,
    ComputeToGraphics,
}

impl AsyncComputeScheduler {
    pub fn schedule(
        passes: &[PassNode],
        execution_order: &[usize],
    ) -> Self {
        let mut graphics_timeline = Vec::new();
        let mut compute_timeline = Vec::new();
        let mut sync_points = Vec::new();
        
        for &pass_idx in execution_order {
            let pass = &passes[pass_idx];
            
            match pass.pass_type {
                PassType::Compute(_) if Self::can_async(pass) => {
                    // Check for sync requirements
                    if !graphics_timeline.is_empty() {
                        sync_points.push(SyncPoint {
                            graphics_pass: *graphics_timeline.last().unwrap(),
                            compute_pass: pass.id,
                            direction: SyncDirection::GraphicsToCompute,
                        });
                    }
                    compute_timeline.push(pass.id);
                }
                _ => {
                    // Check for sync from compute
                    if !compute_timeline.is_empty() {
                        sync_points.push(SyncPoint {
                            graphics_pass: pass.id,
                            compute_pass: *compute_timeline.last().unwrap(),
                            direction: SyncDirection::ComputeToGraphics,
                        });
                    }
                    graphics_timeline.push(pass.id);
                }
            }
        }
        
        Self {
            graphics_timeline,
            compute_timeline,
            sync_points,
        }
    }
    
    fn can_async(pass: &PassNode) -> bool {
        // Heuristic: async if pass doesn't touch render attachments
        match &pass.pass_type {
            PassType::Compute(_) => true,
            _ => false,
        }
    }
}
```

### 23.4.3 Resource Lifetime Management

```rust
pub struct ResourceLifetimeManager {
    pool: TransientResourcePool,
    allocations: HashMap<ResourceId, PhysicalResource>,
    frame_number: u64,
}

impl ResourceLifetimeManager {
    pub fn begin_frame(&mut self) {
        self.frame_number += 1;
        self.pool.release_all();
        self.allocations.clear();
    }
    
    pub fn allocate(
        &mut self,
        device: &wgpu::Device,
        resource: &ResourceNode,
    ) -> PhysicalResource {
        if let Some(existing) = &resource.physical {
            return existing.clone();
        }
        
        match &resource.descriptor {
            ResourceDescriptor::Texture(desc) => {
                let (texture, view) = self.pool.acquire_texture(device, desc, self.frame_number);
                let physical = PhysicalResource::Texture { texture, view };
                self.allocations.insert(resource.id, physical.clone());
                physical
            }
            ResourceDescriptor::Buffer(desc) => {
                let buffer = self.pool.acquire_buffer(device, desc, self.frame_number);
                let physical = PhysicalResource::Buffer { buffer };
                self.allocations.insert(resource.id, physical.clone());
                physical
            }
        }
    }
    
    pub fn end_frame(&mut self) {
        self.pool.gc(self.frame_number, 10); // Keep unused resources for 10 frames
    }
}
```

### 23.4.4 Frame-to-Frame Resource Recycling

```rust
pub struct ResourceRecycler {
    recycled_textures: HashMap<TextureKey, Vec<wgpu::Texture>>,
    recycled_buffers: HashMap<BufferKey, Vec<wgpu::Buffer>>,
}

#[derive(Hash, Eq, PartialEq)]
pub struct TextureKey {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
}

impl ResourceRecycler {
    pub fn recycle_texture(&mut self, texture: wgpu::Texture) {
        let key = TextureKey {
            width: texture.width(),
            height: texture.height(),
            format: texture.format(),
            usage: texture.usage(),
        };
        
        self.recycled_textures
            .entry(key)
            .or_default()
            .push(texture);
    }
    
    pub fn acquire_texture(
        &mut self,
        device: &wgpu::Device,
        key: &TextureKey,
    ) -> wgpu::Texture {
        if let Some(textures) = self.recycled_textures.get_mut(key) {
            if let Some(texture) = textures.pop() {
                return texture;
            }
        }
        
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RecycledTexture"),
            size: wgpu::Extent3d {
                width: key.width,
                height: key.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: key.format,
            usage: key.usage,
            view_formats: &[],
        })
    }
}
```

---

# Chapter 24: Python Bridge

TRINITY's Python bridge enables scripting and rapid prototyping through PyO3 bindings to the Rust renderer.

---

## 24.1 PyO3 Binding Layer

### 24.1.1 Type Marshalling

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
#[derive(Clone)]
pub struct PyTextureDescriptor {
    #[pyo3(get, set)]
    pub width: u32,
    #[pyo3(get, set)]
    pub height: u32,
    #[pyo3(get, set)]
    pub format: String,
    #[pyo3(get, set)]
    pub usage: Vec<String>,
}

#[pymethods]
impl PyTextureDescriptor {
    #[new]
    fn new(width: u32, height: u32, format: &str) -> Self {
        Self {
            width,
            height,
            format: format.to_string(),
            usage: vec!["RENDER_ATTACHMENT".to_string()],
        }
    }
    
    fn to_wgpu(&self) -> TextureResourceDesc {
        TextureResourceDesc {
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            format: parse_format(&self.format),
            usage: parse_usage(&self.usage),
            mip_count: 1,
            sample_count: 1,
        }
    }
}

fn parse_format(s: &str) -> wgpu::TextureFormat {
    match s {
        "RGBA8" => wgpu::TextureFormat::Rgba8Unorm,
        "RGBA8_SRGB" => wgpu::TextureFormat::Rgba8UnormSrgb,
        "BGRA8" => wgpu::TextureFormat::Bgra8Unorm,
        "BGRA8_SRGB" => wgpu::TextureFormat::Bgra8UnormSrgb,
        "R32F" => wgpu::TextureFormat::R32Float,
        "RGBA16F" => wgpu::TextureFormat::Rgba16Float,
        "RGBA32F" => wgpu::TextureFormat::Rgba32Float,
        "DEPTH32F" => wgpu::TextureFormat::Depth32Float,
        "DEPTH24_STENCIL8" => wgpu::TextureFormat::Depth24PlusStencil8,
        _ => panic!("Unknown format: {}", s),
    }
}

fn parse_usage(usage: &[String]) -> wgpu::TextureUsages {
    let mut result = wgpu::TextureUsages::empty();
    for u in usage {
        result |= match u.as_str() {
            "RENDER_ATTACHMENT" => wgpu::TextureUsages::RENDER_ATTACHMENT,
            "TEXTURE_BINDING" => wgpu::TextureUsages::TEXTURE_BINDING,
            "STORAGE_BINDING" => wgpu::TextureUsages::STORAGE_BINDING,
            "COPY_SRC" => wgpu::TextureUsages::COPY_SRC,
            "COPY_DST" => wgpu::TextureUsages::COPY_DST,
            _ => panic!("Unknown usage: {}", u),
        };
    }
    result
}
```

### 24.1.2 Handle Management

```rust
use std::sync::{Arc, Mutex};

#[pyclass]
pub struct PyResourceHandle {
    id: ResourceId,
    name: String,
    resource_type: String,
}

#[pyclass]
pub struct PyRenderer {
    inner: Arc<Mutex<TrinityRenderer>>,
}

#[pymethods]
impl PyRenderer {
    #[new]
    fn new() -> PyResult<Self> {
        let renderer = TrinityRenderer::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self {
            inner: Arc::new(Mutex::new(renderer)),
        })
    }
    
    fn create_texture(&self, desc: &PyTextureDescriptor) -> PyResult<PyResourceHandle> {
        let mut renderer = self.inner.lock().unwrap();
        let id = renderer.create_texture(&desc.to_wgpu());
        
        Ok(PyResourceHandle {
            id,
            name: format!("Texture_{}", id.0),
            resource_type: "texture".to_string(),
        })
    }
    
    fn destroy_resource(&self, handle: &PyResourceHandle) -> PyResult<()> {
        let mut renderer = self.inner.lock().unwrap();
        renderer.destroy_resource(handle.id);
        Ok(())
    }
}
```

### 24.1.3 Callback Patterns

```rust
#[pyclass]
pub struct PyRenderCallback {
    callback: PyObject,
}

#[pymethods]
impl PyRenderCallback {
    #[new]
    fn new(callback: PyObject) -> Self {
        Self { callback }
    }
}

impl PyRenderer {
    fn render_frame_with_callback(&self, py: Python, callback: &PyRenderCallback) -> PyResult<()> {
        let mut renderer = self.inner.lock().unwrap();
        
        // Begin frame
        renderer.begin_frame();
        
        // Call Python callback to set up passes
        let ctx = PyRenderContext::new(&renderer);
        callback.callback.call1(py, (ctx,))?;
        
        // Execute frame
        renderer.execute_frame();
        
        Ok(())
    }
}

#[pyclass]
pub struct PyRenderContext {
    passes: Vec<PyPassBuilder>,
}

#[pymethods]
impl PyRenderContext {
    fn add_render_pass(&mut self, name: &str, config: PyDict) -> PyResult<PyPassBuilder> {
        let builder = PyPassBuilder::new(name, PassType::Render);
        self.passes.push(builder.clone());
        Ok(builder)
    }
    
    fn add_compute_pass(&mut self, name: &str) -> PyResult<PyPassBuilder> {
        let builder = PyPassBuilder::new(name, PassType::Compute);
        self.passes.push(builder.clone());
        Ok(builder)
    }
}
```

### 24.1.4 Error Propagation

```rust
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyIOError};

pub fn wgpu_error_to_py(error: wgpu::Error) -> PyErr {
    match error {
        wgpu::Error::OutOfMemory { .. } => {
            PyErr::new::<PyRuntimeError, _>("GPU out of memory")
        }
        wgpu::Error::Validation { description, .. } => {
            PyErr::new::<PyValueError, _>(format!("Validation error: {}", description))
        }
        wgpu::Error::Internal { description, .. } => {
            PyErr::new::<PyRuntimeError, _>(format!("Internal error: {}", description))
        }
    }
}

pub fn surface_error_to_py(error: wgpu::SurfaceError) -> PyErr {
    match error {
        wgpu::SurfaceError::Timeout => {
            PyErr::new::<PyRuntimeError, _>("Surface timeout")
        }
        wgpu::SurfaceError::Outdated => {
            PyErr::new::<PyRuntimeError, _>("Surface outdated - resize needed")
        }
        wgpu::SurfaceError::Lost => {
            PyErr::new::<PyRuntimeError, _>("Surface lost - recreation needed")
        }
        wgpu::SurfaceError::OutOfMemory => {
            PyErr::new::<PyRuntimeError, _>("Out of memory")
        }
    }
}
```

---

## 24.2 Resource Descriptors

### 24.2.1 Python-Side Descriptors

```python
# trinity/descriptors.py

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class TextureFormat(Enum):
    RGBA8 = "RGBA8"
    RGBA8_SRGB = "RGBA8_SRGB"
    BGRA8 = "BGRA8"
    R32F = "R32F"
    RGBA16F = "RGBA16F"
    RGBA32F = "RGBA32F"
    DEPTH32F = "DEPTH32F"
    DEPTH24_STENCIL8 = "DEPTH24_STENCIL8"

class TextureUsage(Enum):
    RENDER_ATTACHMENT = "RENDER_ATTACHMENT"
    TEXTURE_BINDING = "TEXTURE_BINDING"
    STORAGE_BINDING = "STORAGE_BINDING"
    COPY_SRC = "COPY_SRC"
    COPY_DST = "COPY_DST"

@dataclass
class TextureDesc:
    width: int
    height: int
    format: TextureFormat = TextureFormat.RGBA8
    usage: List[TextureUsage] = None
    mip_levels: int = 1
    sample_count: int = 1
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = [TextureUsage.RENDER_ATTACHMENT]
    
    def to_native(self):
        from trinity._native import PyTextureDescriptor
        desc = PyTextureDescriptor(self.width, self.height, self.format.value)
        desc.usage = [u.value for u in self.usage]
        return desc

@dataclass
class BufferDesc:
    size: int
    usage: List[str]
    mapped_at_creation: bool = False
    
    def to_native(self):
        from trinity._native import PyBufferDescriptor
        return PyBufferDescriptor(self.size, self.usage)
```

### 24.2.2 Descriptor Validation

```rust
impl PyTextureDescriptor {
    fn validate(&self) -> PyResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Texture dimensions must be non-zero"
            ));
        }
        
        if self.width > 16384 || self.height > 16384 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Texture dimensions exceed maximum (16384)"
            ));
        }
        
        // Validate format string
        let valid_formats = [
            "RGBA8", "RGBA8_SRGB", "BGRA8", "BGRA8_SRGB",
            "R32F", "RGBA16F", "RGBA32F",
            "DEPTH32F", "DEPTH24_STENCIL8"
        ];
        
        if !valid_formats.contains(&self.format.as_str()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid texture format: {}", self.format)
            ));
        }
        
        Ok(())
    }
}
```

### 24.2.3 Descriptor to wgpu Translation

```rust
impl PyTextureDescriptor {
    fn to_wgpu_descriptor(&self) -> wgpu::TextureDescriptor<'static> {
        wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: parse_format(&self.format),
            usage: parse_usage(&self.usage),
            view_formats: &[],
        }
    }
}

impl PyBufferDescriptor {
    fn to_wgpu_descriptor(&self) -> wgpu::BufferDescriptor<'static> {
        wgpu::BufferDescriptor {
            label: None,
            size: self.size,
            usage: self.parse_usage(),
            mapped_at_creation: self.mapped_at_creation,
        }
    }
    
    fn parse_usage(&self) -> wgpu::BufferUsages {
        let mut result = wgpu::BufferUsages::empty();
        for u in &self.usage {
            result |= match u.as_str() {
                "VERTEX" => wgpu::BufferUsages::VERTEX,
                "INDEX" => wgpu::BufferUsages::INDEX,
                "UNIFORM" => wgpu::BufferUsages::UNIFORM,
                "STORAGE" => wgpu::BufferUsages::STORAGE,
                "INDIRECT" => wgpu::BufferUsages::INDIRECT,
                "COPY_SRC" => wgpu::BufferUsages::COPY_SRC,
                "COPY_DST" => wgpu::BufferUsages::COPY_DST,
                "MAP_READ" => wgpu::BufferUsages::MAP_READ,
                "MAP_WRITE" => wgpu::BufferUsages::MAP_WRITE,
                _ => panic!("Unknown buffer usage: {}", u),
            };
        }
        result
    }
}
```

### 24.2.4 Descriptor Caching

```rust
pub struct DescriptorCache {
    texture_cache: HashMap<u64, wgpu::Texture>,
    buffer_cache: HashMap<u64, wgpu::Buffer>,
    pipeline_cache: HashMap<u64, wgpu::RenderPipeline>,
}

impl DescriptorCache {
    pub fn get_or_create_texture(
        &mut self,
        device: &wgpu::Device,
        desc: &PyTextureDescriptor,
    ) -> &wgpu::Texture {
        let hash = self.hash_texture_desc(desc);
        
        self.texture_cache.entry(hash).or_insert_with(|| {
            device.create_texture(&desc.to_wgpu_descriptor())
        })
    }
    
    fn hash_texture_desc(&self, desc: &PyTextureDescriptor) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        desc.width.hash(&mut hasher);
        desc.height.hash(&mut hasher);
        desc.format.hash(&mut hasher);
        for u in &desc.usage {
            u.hash(&mut hasher);
        }
        hasher.finish()
    }
}
```

---

## 24.3 Command Recording

### 24.3.1 Python Command Builder

```python
# trinity/commands.py

class RenderPassBuilder:
    def __init__(self, name: str):
        self.name = name
        self.color_attachments = []
        self.depth_attachment = None
        self.commands = []
    
    def add_color_attachment(
        self,
        texture: "TextureHandle",
        load_op: str = "clear",
        clear_color: tuple = (0, 0, 0, 1)
    ):
        self.color_attachments.append({
            "texture": texture,
            "load_op": load_op,
            "clear_color": clear_color
        })
        return self
    
    def set_depth_attachment(
        self,
        texture: "TextureHandle",
        load_op: str = "clear",
        clear_depth: float = 1.0
    ):
        self.depth_attachment = {
            "texture": texture,
            "load_op": load_op,
            "clear_depth": clear_depth
        }
        return self
    
    def set_pipeline(self, pipeline: "PipelineHandle"):
        self.commands.append(("set_pipeline", pipeline))
        return self
    
    def set_bind_group(self, index: int, bind_group: "BindGroupHandle"):
        self.commands.append(("set_bind_group", index, bind_group))
        return self
    
    def set_vertex_buffer(self, slot: int, buffer: "BufferHandle"):
        self.commands.append(("set_vertex_buffer", slot, buffer))
        return self
    
    def set_index_buffer(self, buffer: "BufferHandle", format: str = "uint32"):
        self.commands.append(("set_index_buffer", buffer, format))
        return self
    
    def draw(self, vertex_count: int, instance_count: int = 1):
        self.commands.append(("draw", vertex_count, instance_count, 0, 0))
        return self
    
    def draw_indexed(self, index_count: int, instance_count: int = 1):
        self.commands.append(("draw_indexed", index_count, instance_count, 0, 0, 0))
        return self


class ComputePassBuilder:
    def __init__(self, name: str):
        self.name = name
        self.commands = []
    
    def set_pipeline(self, pipeline: "ComputePipelineHandle"):
        self.commands.append(("set_pipeline", pipeline))
        return self
    
    def set_bind_group(self, index: int, bind_group: "BindGroupHandle"):
        self.commands.append(("set_bind_group", index, bind_group))
        return self
    
    def dispatch(self, x: int, y: int = 1, z: int = 1):
        self.commands.append(("dispatch", x, y, z))
        return self
```

### 24.3.2 Deferred Execution

```rust
#[pyclass]
pub struct PyCommandList {
    commands: Vec<RecordedCommand>,
}

pub enum RecordedCommand {
    BeginRenderPass(RenderPassConfig),
    EndRenderPass,
    SetPipeline(PipelineId),
    SetBindGroup(u32, BindGroupId, Vec<u32>),
    SetVertexBuffer(u32, BufferId, u64),
    SetIndexBuffer(BufferId, wgpu::IndexFormat, u64),
    Draw(u32, u32, u32, u32),
    DrawIndexed(u32, u32, u32, i32, u32),
    BeginComputePass,
    EndComputePass,
    Dispatch(u32, u32, u32),
}

impl PyCommandList {
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, resources: &ResourceManager) {
        let mut current_render_pass: Option<wgpu::RenderPass> = None;
        let mut current_compute_pass: Option<wgpu::ComputePass> = None;
        
        for cmd in &self.commands {
            match cmd {
                RecordedCommand::BeginRenderPass(config) => {
                    let pass = encoder.begin_render_pass(&config.to_wgpu(resources));
                    current_render_pass = Some(pass);
                }
                RecordedCommand::EndRenderPass => {
                    current_render_pass = None;
                }
                RecordedCommand::SetPipeline(id) => {
                    if let Some(ref mut pass) = current_render_pass {
                        pass.set_pipeline(resources.get_render_pipeline(*id));
                    }
                    if let Some(ref mut pass) = current_compute_pass {
                        pass.set_pipeline(resources.get_compute_pipeline(*id));
                    }
                }
                RecordedCommand::Draw(verts, instances, first_vert, first_inst) => {
                    if let Some(ref mut pass) = current_render_pass {
                        pass.draw(*verts..*verts + *instances, *first_vert..*first_inst);
                    }
                }
                // ... handle other commands
                _ => {}
            }
        }
    }
}
```

### 24.3.3 Command Batching

```rust
impl PyCommandList {
    pub fn optimize(&mut self) {
        // Remove redundant state changes
        self.remove_redundant_pipeline_sets();
        self.remove_redundant_bind_groups();
        
        // Batch compatible draw calls
        self.batch_draws();
    }
    
    fn remove_redundant_pipeline_sets(&mut self) {
        let mut last_pipeline: Option<PipelineId> = None;
        
        self.commands.retain(|cmd| {
            if let RecordedCommand::SetPipeline(id) = cmd {
                if Some(*id) == last_pipeline {
                    return false;
                }
                last_pipeline = Some(*id);
            }
            true
        });
    }
    
    fn batch_draws(&mut self) {
        // Group consecutive draws with same state into multi-draw
        // Implementation depends on multi-draw support
    }
}
```

### 24.3.4 Error Handling

```python
# trinity/errors.py

class TrinityError(Exception):
    pass

class ValidationError(TrinityError):
    pass

class OutOfMemoryError(TrinityError):
    pass

class DeviceLostError(TrinityError):
    pass

def handle_native_error(error):
    error_str = str(error)
    
    if "Validation" in error_str:
        raise ValidationError(error_str)
    elif "OutOfMemory" in error_str:
        raise OutOfMemoryError(error_str)
    elif "DeviceLost" in error_str:
        raise DeviceLostError(error_str)
    else:
        raise TrinityError(error_str)
```

---

# Complete Python API Example

```python
# example.py

from trinity import Renderer, TextureDesc, TextureFormat, TextureUsage
from trinity import RenderPassBuilder, ComputePassBuilder
from trinity.math import Vec3, Mat4

def main():
    # Initialize renderer
    renderer = Renderer()
    
    # Create resources
    color_target = renderer.create_texture(TextureDesc(
        width=1920,
        height=1080,
        format=TextureFormat.RGBA8_SRGB,
        usage=[TextureUsage.RENDER_ATTACHMENT, TextureUsage.TEXTURE_BINDING]
    ))
    
    depth_target = renderer.create_texture(TextureDesc(
        width=1920,
        height=1080,
        format=TextureFormat.DEPTH32F,
        usage=[TextureUsage.RENDER_ATTACHMENT]
    ))
    
    # Load shader and create pipeline
    shader = renderer.create_shader_module("shaders/pbr.wgsl")
    pipeline = renderer.create_render_pipeline(
        shader=shader,
        vertex_format=["position", "normal", "uv"],
        color_format=TextureFormat.RGBA8_SRGB,
        depth_format=TextureFormat.DEPTH32F
    )
    
    # Main render loop
    while renderer.window_open():
        # Build frame
        frame = renderer.begin_frame()
        
        # Shadow pass
        shadow_pass = RenderPassBuilder("Shadows")
        shadow_pass.set_depth_attachment(shadow_map, load_op="clear")
        shadow_pass.set_pipeline(shadow_pipeline)
        for obj in scene.shadow_casters:
            shadow_pass.draw_mesh(obj.mesh, obj.transform)
        frame.add_pass(shadow_pass)
        
        # Main pass
        main_pass = RenderPassBuilder("Main")
        main_pass.add_color_attachment(color_target, clear_color=(0.1, 0.1, 0.1, 1))
        main_pass.set_depth_attachment(depth_target)
        main_pass.set_pipeline(pipeline)
        for obj in scene.objects:
            main_pass.draw_mesh(obj.mesh, obj.transform, obj.material)
        frame.add_pass(main_pass)
        
        # Post-processing
        post_pass = ComputePassBuilder("PostProcess")
        post_pass.set_pipeline(tonemap_pipeline)
        post_pass.set_bind_group(0, tonemap_bindings)
        post_pass.dispatch(1920 // 8, 1080 // 8, 1)
        frame.add_pass(post_pass)
        
        # Execute and present
        renderer.execute_frame(frame)
        renderer.present()

if __name__ == "__main__":
    main()
```

---

# TRINITY Integration Summary

| Component | Rust Module | Python Module | Purpose |
|-----------|-------------|---------------|---------|
| Frame Graph | `frame_graph` | `trinity.graph` | Pass scheduling & resources |
| Resource Pool | `resource_pool` | Internal | Transient allocation |
| Barrier System | `barriers` | Internal | Automatic sync |
| Command Builder | `commands` | `trinity.commands` | GPU command recording |
| Type Bindings | `pyo3_bindings` | `trinity._native` | Rust-Python bridge |
| Error Handling | `errors` | `trinity.errors` | Cross-language errors |

---

*End of WGPU_PART_XII_INTEGRATION.md*
