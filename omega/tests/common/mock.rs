//! Mock/headless RHI backend for integration tests.
//!
//! All types mirror the Python Null* implementations in engine/platform/rhi/.
//! Thread-safe where needed (fences, swapchain indices).
//! Handle generation uses atomic counters matching Python's threading.Lock pattern.

// Allow dead code: different test files use different subsets of these types.
#![allow(dead_code)]

use bitflags::bitflags;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const NULL_VENDOR_ID: u32 = 0x0000;
pub const NULL_DEVICE_ID: u32 = 0x0000;
pub const DEFAULT_DISCRETE_VRAM: u64 = 8 * 1024 * 1024 * 1024; // 8 GB
pub const DEFAULT_SHARED_MEMORY: u64 = 16 * 1024 * 1024 * 1024; // 16 GB
pub const DEFAULT_MAX_TEXTURE_SIZE: u32 = 16384;
pub const DEFAULT_MAX_BUFFER_SIZE: u64 = 4 * 1024 * 1024 * 1024; // 4 GB
pub const DEFAULT_MAX_LOD: f32 = 1000.0;

// ---------------------------------------------------------------------------
// Handle generation (atomic counters mirroring Python class-level locks)
// ---------------------------------------------------------------------------

static NEXT_BUFFER_HANDLE: AtomicU64 = AtomicU64::new(BUFFER_HANDLE_START);
static NEXT_TEXTURE_HANDLE: AtomicU64 = AtomicU64::new(TEXTURE_HANDLE_START);
static NEXT_SAMPLER_HANDLE: AtomicU64 = AtomicU64::new(SAMPLER_HANDLE_START);
static NEXT_SHADER_HANDLE: AtomicU64 = AtomicU64::new(SHADER_HANDLE_START);
static NEXT_PIPELINE_HANDLE: AtomicU64 = AtomicU64::new(PIPELINE_HANDLE_START);
static NEXT_SWAPCHAIN_HANDLE: AtomicU64 = AtomicU64::new(SWAPCHAIN_HANDLE_START);
static NEXT_FENCE_HANDLE: AtomicU64 = AtomicU64::new(FENCE_HANDLE_START);
static NEXT_HEAP_INDEX: AtomicU64 = AtomicU64::new(HEAP_HANDLE_START);

pub const BUFFER_HANDLE_START: u64 = 1;
pub const TEXTURE_HANDLE_START: u64 = 1000;
pub const SAMPLER_HANDLE_START: u64 = 2000;
pub const SHADER_HANDLE_START: u64 = 3000;
pub const PIPELINE_HANDLE_START: u64 = 4000;
pub const SWAPCHAIN_HANDLE_START: u64 = 5000;
pub const FENCE_HANDLE_START: u64 = 6000;
pub const HEAP_HANDLE_START: u64 = 7000;

fn next_handle(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::SeqCst)
}

fn reset_handle_counters() {
    NEXT_BUFFER_HANDLE.store(BUFFER_HANDLE_START, Ordering::SeqCst);
    NEXT_TEXTURE_HANDLE.store(TEXTURE_HANDLE_START, Ordering::SeqCst);
    NEXT_SAMPLER_HANDLE.store(SAMPLER_HANDLE_START, Ordering::SeqCst);
    NEXT_SHADER_HANDLE.store(SHADER_HANDLE_START, Ordering::SeqCst);
    NEXT_PIPELINE_HANDLE.store(PIPELINE_HANDLE_START, Ordering::SeqCst);
    NEXT_SWAPCHAIN_HANDLE.store(SWAPCHAIN_HANDLE_START, Ordering::SeqCst);
    NEXT_FENCE_HANDLE.store(FENCE_HANDLE_START, Ordering::SeqCst);
    NEXT_HEAP_INDEX.store(HEAP_HANDLE_START, Ordering::SeqCst);
}

// ---------------------------------------------------------------------------
// Enums — mirror engine/platform/rhi/*.py exactly
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdapterType {
    Discrete,
    Integrated,
    Software,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueType {
    Graphics,
    Compute,
    Transfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Default,
    Upload,
    Readback,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BufferUsage: u32 {
        const VERTEX    = 1 << 0;
        const INDEX     = 1 << 1;
        const CONSTANT  = 1 << 2;
        const STORAGE   = 1 << 3;
        const INDIRECT  = 1 << 4;
        const COPY_SRC  = 1 << 5;
        const COPY_DST  = 1 << 6;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    R8Unorm,
    RG8Unorm,
    RGBA8Unorm,
    RGBA16Float,
    RGBA32Float,
    R32Float,
    R32Uint,
    R16Uint,
    D32Float,
    D24S8,
    BC7Unorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureType {
    Texture1D,
    Texture2D,
    Texture3D,
    TextureCube,
    TextureArray,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TextureUsage: u32 {
        const SHADER_RESOURCE  = 1 << 0;
        const RENDER_TARGET    = 1 << 1;
        const DEPTH_STENCIL    = 1 << 2;
        const UNORDERED_ACCESS = 1 << 3;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleCount {
    X1 = 1,
    X2 = 2,
    X4 = 4,
    X8 = 8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    Wrap,
    Clamp,
    Mirror,
    Border,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Pixel,
    Compute,
    Hull,
    Domain,
    Geometry,
    Mesh,
    Task,
    RayGeneration,
    Miss,
    ClosestHit,
    AnyHit,
    Intersection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    Solid,
    Wireframe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    InvSrcColor,
    SrcAlpha,
    InvSrcAlpha,
    DstColor,
    InvDstColor,
    DstAlpha,
    InvDstAlpha,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendOp {
    Add,
    Subtract,
    RevSubtract,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineType {
    Graphics,
    Compute,
    Raytracing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceState {
    Undefined,
    Common,
    RenderTarget,
    DepthWrite,
    DepthRead,
    ShaderResource,
    UnorderedAccess,
    CopySrc,
    CopyDst,
    Present,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierType {
    Transition,
    Uav,
    Aliasing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentMode {
    Immediate,
    Vsync,
    Mailbox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb,
    ScRgb,
    Hdr10,
    Pq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorType {
    Cbv,
    Srv,
    Uav,
    Sampler,
}

// ---------------------------------------------------------------------------
// Descriptor / info structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub dedicated_video_memory: u64,
    pub dedicated_system_memory: u64,
    pub shared_system_memory: u64,
    pub adapter_type: AdapterType,
    pub vendor_id: u32,
    pub device_id: u32,
}

#[derive(Debug, Clone)]
pub struct FeatureSupport {
    pub ray_tracing: bool,
    pub mesh_shaders: bool,
    pub bindless: bool,
    pub compute: bool,
    pub max_texture_size: u32,
    pub max_buffer_size: u64,
}

#[derive(Debug, Clone)]
pub struct FormatSupport {
    pub renderable: bool,
    pub filterable: bool,
    pub blendable: bool,
    pub storage: bool,
    pub multisample: bool,
}

#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub enable_debug: bool,
    pub enable_validation: bool,
}

#[derive(Debug, Clone)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsage,
    pub memory_type: MemoryType,
    pub stride: u32,
}

#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub ty: TextureType,
    pub format: Format,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub array_size: u32,
    pub sample_count: SampleCount,
    pub usage: TextureUsage,
}

#[derive(Debug, Clone)]
pub struct SamplerDesc {
    pub min_filter: FilterMode,
    pub mag_filter: FilterMode,
    pub mip_filter: FilterMode,
    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
    pub compare_op: Option<CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            address_u: AddressMode::Wrap,
            address_v: AddressMode::Wrap,
            address_w: AddressMode::Wrap,
            mip_lod_bias: 0.0,
            max_anisotropy: 1,
            compare_op: None,
            min_lod: 0.0,
            max_lod: DEFAULT_MAX_LOD,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShaderDesc {
    pub stage: ShaderStage,
    pub source: Vec<u8>,
    pub entry_point: String,
}

#[derive(Debug, Clone)]
pub struct RasterizerState {
    pub fill_mode: FillMode,
    pub cull_mode: CullMode,
    pub front_ccw: bool,
    pub depth_bias: i32,
    pub depth_clip: bool,
}

impl Default for RasterizerState {
    fn default() -> Self {
        Self {
            fill_mode: FillMode::Solid,
            cull_mode: CullMode::Back,
            front_ccw: false,
            depth_bias: 0,
            depth_clip: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DepthStencilState {
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_func: CompareOp,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_test: true,
            depth_write: true,
            depth_func: CompareOp::Less,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlendState {
    pub enabled: bool,
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOp,
}

impl Default for BlendState {
    fn default() -> Self {
        Self {
            enabled: false,
            src_color: BlendFactor::One,
            dst_color: BlendFactor::Zero,
            color_op: BlendOp::Add,
            src_alpha: BlendFactor::One,
            dst_alpha: BlendFactor::Zero,
            alpha_op: BlendOp::Add,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphicsPipelineDesc {
    pub vertex_shader: Option<ShaderDesc>,
    pub pixel_shader: Option<ShaderDesc>,
    pub geometry_shader: Option<ShaderDesc>,
    pub hull_shader: Option<ShaderDesc>,
    pub domain_shader: Option<ShaderDesc>,
    pub topology: PrimitiveTopology,
    pub rasterizer: RasterizerState,
    pub depth_stencil: DepthStencilState,
    pub blend: BlendState,
    pub render_target_formats: Vec<Format>,
    pub depth_format: Option<Format>,
}

impl Default for GraphicsPipelineDesc {
    fn default() -> Self {
        Self {
            vertex_shader: None,
            pixel_shader: None,
            geometry_shader: None,
            hull_shader: None,
            domain_shader: None,
            topology: PrimitiveTopology::TriangleList,
            rasterizer: RasterizerState::default(),
            depth_stencil: DepthStencilState::default(),
            blend: BlendState::default(),
            render_target_formats: Vec::new(),
            depth_format: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComputePipelineDesc {
    pub compute_shader: Option<ShaderDesc>,
}

#[derive(Debug, Clone)]
pub struct RaytracingPipelineDesc {
    pub ray_gen_shader: Option<ShaderDesc>,
    pub miss_shaders: Vec<ShaderDesc>,
    pub hit_groups: Vec<String>,
    pub max_recursion_depth: u32,
}

#[derive(Debug, Clone)]
pub struct SwapchainDesc {
    pub width: u32,
    pub height: u32,
    pub format: Format,
    pub buffer_count: u32,
    pub present_mode: PresentMode,
    pub color_space: ColorSpace,
}

#[derive(Debug, Clone)]
pub struct BarrierDesc {
    pub ty: BarrierType,
    pub resource: Option<u64>, // handle stored as u64
    pub state_before: ResourceState,
    pub state_after: ResourceState,
}

#[derive(Debug, Clone, Copy)]
pub struct DescriptorHandle {
    pub heap_index: u64,
    pub offset: u64,
}

// ---------------------------------------------------------------------------
// Recorded command — mirrors Python's Command(type, args) dataclass
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RecordedCommand {
    pub cmd_type: String,
    pub args: std::collections::HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Mock Resource Objects
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MockBuffer {
    handle: u64,
    desc: BufferDesc,
    valid: bool,
}

impl MockBuffer {
    pub fn new(desc: BufferDesc) -> Self {
        Self {
            handle: next_handle(&NEXT_BUFFER_HANDLE),
            desc,
            valid: true,
        }
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }

    pub fn desc(&self) -> &BufferDesc {
        &self.desc
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn destroy(&mut self) {
        self.valid = false;
    }
}

#[derive(Debug, Clone)]
pub struct MockTexture {
    handle: u64,
    desc: TextureDesc,
    valid: bool,
}

impl MockTexture {
    pub fn new(desc: TextureDesc) -> Self {
        Self {
            handle: next_handle(&NEXT_TEXTURE_HANDLE),
            desc,
            valid: true,
        }
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }

    pub fn desc(&self) -> &TextureDesc {
        &self.desc
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn destroy(&mut self) {
        self.valid = false;
    }
}

#[derive(Debug, Clone)]
pub struct MockSampler {
    handle: u64,
    desc: SamplerDesc,
    valid: bool,
}

impl MockSampler {
    pub fn new(desc: SamplerDesc) -> Self {
        Self {
            handle: next_handle(&NEXT_SAMPLER_HANDLE),
            desc,
            valid: true,
        }
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }

    pub fn desc(&self) -> &SamplerDesc {
        &self.desc
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn destroy(&mut self) {
        self.valid = false;
    }
}

// ---------------------------------------------------------------------------
// Mock Pipeline
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MockShader {
    handle: u64,
    desc: ShaderDesc,
    valid: bool,
}

impl MockShader {
    pub fn new(desc: ShaderDesc) -> Self {
        Self {
            handle: next_handle(&NEXT_SHADER_HANDLE),
            desc,
            valid: true,
        }
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }

    pub fn desc(&self) -> &ShaderDesc {
        &self.desc
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }
}

#[derive(Debug, Clone)]
pub struct MockPipelineState {
    handle: u64,
    desc_serialized: String, // serialized desc for storage
    pipeline_ty: PipelineType,
    valid: bool,
}

impl MockPipelineState {
    pub fn new_graphics(desc: &GraphicsPipelineDesc) -> Self {
        Self {
            handle: next_handle(&NEXT_PIPELINE_HANDLE),
            desc_serialized: format!("{:?}", desc),
            pipeline_ty: PipelineType::Graphics,
            valid: true,
        }
    }

    pub fn new_compute(desc: &ComputePipelineDesc) -> Self {
        Self {
            handle: next_handle(&NEXT_PIPELINE_HANDLE),
            desc_serialized: format!("{:?}", desc),
            pipeline_ty: PipelineType::Compute,
            valid: true,
        }
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }

    pub fn pipeline_type(&self) -> PipelineType {
        self.pipeline_ty
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }
}

// ---------------------------------------------------------------------------
// Mock Command List
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MockCommandList {
    recording: bool,
    commands: Vec<RecordedCommand>,
}

impl MockCommandList {
    pub fn new() -> Self {
        Self {
            recording: false,
            commands: Vec::new(),
        }
    }

    pub fn begin(&mut self) {
        self.recording = true;
        self.commands.clear();
    }

    pub fn end(&mut self) {
        self.recording = false;
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    fn record(&mut self, cmd_type: &str, args: Vec<(&str, String)>) {
        if !self.recording {
            return;
        }
        let mut map = std::collections::HashMap::new();
        for (k, v) in args {
            map.insert(k.to_string(), v);
        }
        self.commands.push(RecordedCommand {
            cmd_type: cmd_type.to_string(),
            args: map,
        });
    }

    pub fn barrier(&mut self, resource_handle: u64, state_before: ResourceState, state_after: ResourceState) {
        self.record(
            "barrier",
            vec![
                ("resource", format!("{}", resource_handle)),
                ("state_before", format!("{:?}", state_before)),
                ("state_after", format!("{:?}", state_after)),
            ],
        );
    }

    pub fn begin_render_pass(
        &mut self,
        render_target_handles: &[u64],
        depth_target_handle: Option<u64>,
        clear_color: Option<(f32, f32, f32, f32)>,
        clear_depth: Option<f32>,
    ) {
        let mut args = vec![
            ("render_targets", format!("{:?}", render_target_handles)),
        ];
        if let Some(dt) = depth_target_handle {
            args.push(("depth_target", format!("{}", dt)));
        }
        if let Some(cc) = clear_color {
            args.push(("clear_color", format!("{:?}", cc)));
        }
        if let Some(cd) = clear_depth {
            args.push(("clear_depth", format!("{}", cd)));
        }
        self.record("begin_render_pass", args);
    }

    pub fn end_render_pass(&mut self) {
        self.record("end_render_pass", vec![]);
    }

    pub fn set_pipeline(&mut self, pipeline_handle: u64) {
        self.record("set_pipeline", vec![("pipeline", format!("{}", pipeline_handle))]);
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        self.record(
            "set_viewport",
            vec![
                ("x", format!("{}", x)),
                ("y", format!("{}", y)),
                ("w", format!("{}", w)),
                ("h", format!("{}", h)),
                ("min_depth", format!("{}", min_depth)),
                ("max_depth", format!("{}", max_depth)),
            ],
        );
    }

    pub fn set_scissor(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.record(
            "set_scissor",
            vec![
                ("x", format!("{}", x)),
                ("y", format!("{}", y)),
                ("w", format!("{}", w)),
                ("h", format!("{}", h)),
            ],
        );
    }

    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_handle: u64, offset: u64, stride: u32) {
        self.record(
            "set_vertex_buffer",
            vec![
                ("slot", format!("{}", slot)),
                ("buffer", format!("{}", buffer_handle)),
                ("offset", format!("{}", offset)),
                ("stride", format!("{}", stride)),
            ],
        );
    }

    pub fn set_index_buffer(&mut self, buffer_handle: u64, offset: u64, format: Format) {
        self.record(
            "set_index_buffer",
            vec![
                ("buffer", format!("{}", buffer_handle)),
                ("offset", format!("{}", offset)),
                ("format", format!("{:?}", format)),
            ],
        );
    }

    pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        self.record(
            "draw",
            vec![
                ("vertex_count", format!("{}", vertex_count)),
                ("instance_count", format!("{}", instance_count)),
                ("first_vertex", format!("{}", first_vertex)),
                ("first_instance", format!("{}", first_instance)),
            ],
        );
    }

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.record(
            "draw_indexed",
            vec![
                ("index_count", format!("{}", index_count)),
                ("instance_count", format!("{}", instance_count)),
                ("first_index", format!("{}", first_index)),
                ("vertex_offset", format!("{}", vertex_offset)),
                ("first_instance", format!("{}", first_instance)),
            ],
        );
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.record(
            "dispatch",
            vec![
                ("x", format!("{}", x)),
                ("y", format!("{}", y)),
                ("z", format!("{}", z)),
            ],
        );
    }

    pub fn copy_buffer(&mut self, dst_handle: u64, dst_offset: u64, src_handle: u64, src_offset: u64, size: u64) {
        self.record(
            "copy_buffer",
            vec![
                ("dst", format!("{}", dst_handle)),
                ("dst_offset", format!("{}", dst_offset)),
                ("src", format!("{}", src_handle)),
                ("src_offset", format!("{}", src_offset)),
                ("size", format!("{}", size)),
            ],
        );
    }

    pub fn recorded_commands(&self) -> &[RecordedCommand] {
        &self.commands
    }
}

// ---------------------------------------------------------------------------
// Mock Queue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct QueueState {
    submitted: Vec<Vec<RecordedCommand>>,
    shutdown: bool,
}

#[derive(Debug)]
pub struct MockQueue {
    queue_type: QueueType,
    state: Arc<Mutex<QueueState>>,
}

impl MockQueue {
    pub fn new(queue_type: QueueType) -> Self {
        Self {
            queue_type,
            state: Arc::new(Mutex::new(QueueState {
                submitted: Vec::new(),
                shutdown: false,
            })),
        }
    }

    pub fn queue_type(&self) -> QueueType {
        self.queue_type
    }

    pub fn submit(&self, cmd_lists: &[MockCommandList]) {
        let mut state = self.state.lock().unwrap();
        for cmd_list in cmd_lists {
            state.submitted.push(cmd_list.commands.clone());
        }
    }

    pub fn submit_with_fence(&self, cmd_lists: &[MockCommandList], fence: &MockFence) {
        self.submit(cmd_lists);
        fence.signal(fence.value() + 1);
    }

    pub fn wait(&self, _fence: &MockFence) {
        // Mock: nothing to wait for
    }

    pub fn signal(&self, fence: &MockFence) {
        fence.signal(fence.value() + 1);
    }

    pub fn submitted_count(&self) -> usize {
        self.state.lock().unwrap().submitted.len()
    }

    pub fn shutdown(&self) {
        let mut state = self.state.lock().unwrap();
        state.shutdown = true;
    }

    pub fn is_shutdown(&self) -> bool {
        self.state.lock().unwrap().shutdown
    }
}

impl Clone for MockQueue {
    fn clone(&self) -> Self {
        Self {
            queue_type: self.queue_type,
            state: Arc::clone(&self.state),
        }
    }
}

// ---------------------------------------------------------------------------
// Mock Fence
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct FenceState {
    value: u64,
    handle: u64,
}

#[derive(Debug, Clone)]
pub struct MockFence {
    state: Arc<(Mutex<FenceState>, Condvar)>,
}

impl MockFence {
    pub fn new(initial: u64) -> Self {
        Self {
            state: Arc::new((
                Mutex::new(FenceState {
                    value: initial,
                    handle: next_handle(&NEXT_FENCE_HANDLE),
                }),
                Condvar::new(),
            )),
        }
    }

    pub fn handle(&self) -> u64 {
        self.state.0.lock().unwrap().handle
    }

    pub fn value(&self) -> u64 {
        self.state.0.lock().unwrap().value
    }

    pub fn signal(&self, value: u64) {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        state.value = value;
        cvar.notify_all();
    }

    pub fn wait(&self, value: u64, timeout_ms: i64) -> bool {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();

        if timeout_ms < 0 {
            // Infinite wait
            while state.value < value {
                state = cvar.wait(state).unwrap();
            }
            true
        } else {
            let start = Instant::now();
            let timeout = std::time::Duration::from_millis(timeout_ms as u64);
            while state.value < value {
                let elapsed = Instant::now() - start;
                if elapsed >= timeout {
                    return false;
                }
                let remaining = timeout - elapsed;
                let result = cvar.wait_timeout(state, remaining).unwrap();
                state = result.0;
                if result.1.timed_out() && state.value < value {
                    return false;
                }
            }
            true
        }
    }

    pub fn is_complete(&self, value: u64) -> bool {
        self.state.0.lock().unwrap().value >= value
    }
}

// ---------------------------------------------------------------------------
// Mock Swapchain
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct SwapchainState {
    current_index: u32,
    desc: SwapchainDesc,
}

#[derive(Debug, Clone)]
pub struct MockSwapchain {
    state: Arc<Mutex<SwapchainState>>,
    /// Back-buffer textures — only retained for index-based access.
    textures: Arc<Mutex<Vec<MockTexture>>>,
}

impl MockSwapchain {
    pub fn new(desc: SwapchainDesc) -> Self {
        let mut textures = Vec::with_capacity(desc.buffer_count as usize);
        for _ in 0..desc.buffer_count {
            let tex_desc = TextureDesc {
                ty: TextureType::Texture2D,
                format: desc.format,
                width: desc.width,
                height: desc.height,
                depth: 1,
                mip_levels: 1,
                array_size: 1,
                sample_count: SampleCount::X1,
                usage: TextureUsage::RENDER_TARGET,
            };
            textures.push(MockTexture::new(tex_desc));
        }

        Self {
            state: Arc::new(Mutex::new(SwapchainState {
                current_index: 0,
                desc,
            })),
            textures: Arc::new(Mutex::new(textures)),
        }
    }

    pub fn current_texture(&self) -> MockTexture {
        let idx = self.state.lock().unwrap().current_index as usize;
        self.textures.lock().unwrap()[idx].clone()
    }

    pub fn current_index(&self) -> u32 {
        self.state.lock().unwrap().current_index
    }

    pub fn present(&self) {
        let mut s = self.state.lock().unwrap();
        s.current_index = (s.current_index + 1) % s.desc.buffer_count;
    }

    pub fn resize(&self, width: u32, height: u32) {
        let mut s = self.state.lock().unwrap();
        s.desc.width = width;
        s.desc.height = height;

        // Recreate textures
        let mut textures = self.textures.lock().unwrap();
        textures.clear();
        for _ in 0..s.desc.buffer_count {
            let tex_desc = TextureDesc {
                ty: TextureType::Texture2D,
                format: s.desc.format,
                width,
                height,
                depth: 1,
                mip_levels: 1,
                array_size: 1,
                sample_count: SampleCount::X1,
                usage: TextureUsage::RENDER_TARGET,
            };
            textures.push(MockTexture::new(tex_desc));
        }
        s.current_index = 0;
    }

    pub fn buffer_count(&self) -> u32 {
        self.state.lock().unwrap().desc.buffer_count
    }
}

// ---------------------------------------------------------------------------
// Mock Descriptor Heap
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct HeapState {
    next_offset: u64,
    free_list: Vec<u64>,
    count: u64,
}

#[derive(Debug, Clone)]
pub struct MockDescriptorHeap {
    heap_index: u64,
    descriptor_type: DescriptorType,
    state: Arc<Mutex<HeapState>>,
}

impl MockDescriptorHeap {
    pub fn new(descriptor_type: DescriptorType, count: u64) -> Self {
        let heap_index = next_handle(&NEXT_HEAP_INDEX);
        Self {
            heap_index,
            descriptor_type,
            state: Arc::new(Mutex::new(HeapState {
                next_offset: 0,
                free_list: Vec::new(),
                count,
            })),
        }
    }

    pub fn descriptor_type(&self) -> DescriptorType {
        self.descriptor_type
    }

    pub fn allocate(&self) -> Option<DescriptorHandle> {
        let mut s = self.state.lock().unwrap();
        if let Some(offset) = s.free_list.pop() {
            return Some(DescriptorHandle {
                heap_index: self.heap_index,
                offset,
            });
        }
        if s.next_offset < s.count {
            let offset = s.next_offset;
            s.next_offset += 1;
            Some(DescriptorHandle {
                heap_index: self.heap_index,
                offset,
            })
        } else {
            None
        }
    }

    pub fn free(&self, handle: DescriptorHandle) {
        if handle.heap_index == self.heap_index {
            let mut s = self.state.lock().unwrap();
            s.free_list.push(handle.offset);
        }
    }
}

// ---------------------------------------------------------------------------
// Mock Adapter
// ---------------------------------------------------------------------------

static NEXT_ADAPTER_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct MockAdapter {
    id: u64,
    adapter_type: AdapterType,
}

impl MockAdapter {
    pub fn new(adapter_type: AdapterType) -> Self {
        Self {
            id: NEXT_ADAPTER_ID.fetch_add(1, Ordering::SeqCst),
            adapter_type,
        }
    }

    pub fn enumerate() -> Vec<MockAdapter> {
        vec![
            MockAdapter::new(AdapterType::Discrete),
            MockAdapter::new(AdapterType::Integrated),
            MockAdapter::new(AdapterType::Software),
        ]
    }

    pub fn info(&self) -> AdapterInfo {
        AdapterInfo {
            name: format!("Mock Adapter {}", self.id),
            dedicated_video_memory: DEFAULT_DISCRETE_VRAM,
            dedicated_system_memory: 0,
            shared_system_memory: DEFAULT_SHARED_MEMORY,
            adapter_type: self.adapter_type,
            vendor_id: NULL_VENDOR_ID,
            device_id: NULL_DEVICE_ID,
        }
    }

    pub fn query_features(&self) -> FeatureSupport {
        FeatureSupport {
            ray_tracing: self.adapter_type == AdapterType::Discrete,
            mesh_shaders: self.adapter_type == AdapterType::Discrete,
            bindless: true,
            compute: true,
            max_texture_size: DEFAULT_MAX_TEXTURE_SIZE,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
        }
    }

    pub fn query_format_support(&self, _format: Format) -> FormatSupport {
        FormatSupport {
            renderable: true,
            filterable: true,
            blendable: true,
            storage: true,
            multisample: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Mock Device
// ---------------------------------------------------------------------------

pub struct MockDevice {
    adapter: MockAdapter,
    debug: bool,
    validation: bool,
    queues: Mutex<std::collections::HashMap<QueueType, MockQueue>>,
    shutdown_called: bool,
}

impl MockDevice {
    pub fn create(adapter: MockAdapter) -> Self {
        Self {
            adapter,
            debug: false,
            validation: false,
            queues: Mutex::new(std::collections::HashMap::new()),
            shutdown_called: false,
        }
    }

    pub fn create_with_config(adapter: MockAdapter, config: DeviceConfig) -> Self {
        Self {
            adapter,
            debug: config.enable_debug,
            validation: config.enable_validation,
            queues: Mutex::new(std::collections::HashMap::new()),
            shutdown_called: false,
        }
    }

    pub fn adapter(&self) -> &MockAdapter {
        &self.adapter
    }

    pub fn debug_enabled(&self) -> bool {
        self.debug
    }

    pub fn validation_enabled(&self) -> bool {
        self.validation
    }

    pub fn get_queue(&self, queue_type: QueueType) -> MockQueue {
        let mut queues = self.queues.lock().unwrap();
        queues
            .entry(queue_type)
            .or_insert_with(|| MockQueue::new(queue_type))
            .clone()
    }

    pub fn create_buffer(&self, desc: BufferDesc) -> MockBuffer {
        MockBuffer::new(desc)
    }

    pub fn create_texture(&self, desc: TextureDesc) -> MockTexture {
        MockTexture::new(desc)
    }

    pub fn create_sampler(&self, desc: SamplerDesc) -> MockSampler {
        MockSampler::new(desc)
    }

    pub fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> MockPipelineState {
        MockPipelineState::new_graphics(desc)
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> MockPipelineState {
        MockPipelineState::new_compute(desc)
    }

    pub fn wait_idle(&self) {
        // Mock: nothing to wait for
    }

    pub fn shutdown(&mut self) {
        self.shutdown_called = true;
        let queues = self.queues.lock().unwrap();
        for queue in queues.values() {
            queue.shutdown();
        }
    }

    pub fn is_shutdown(&self) -> bool {
        self.shutdown_called
    }
}

/// Reset all atomic counters so each test starts with deterministic handles.
/// Call this at the start of every test that may be sensitive to handle values.
pub fn reset_test_state() {
    reset_handle_counters();
    NEXT_ADAPTER_ID.store(0, Ordering::SeqCst);
}

/// Create a standard test device using a Software-type adapter.
pub fn create_test_device() -> MockDevice {
    reset_test_state();
    let adapter = MockAdapter::new(AdapterType::Software);
    MockDevice::create(adapter)
}

/// Create a test device with discrete adapter (RT/mesh shaders enabled).
pub fn create_discrete_device() -> MockDevice {
    reset_test_state();
    let adapter = MockAdapter::new(AdapterType::Discrete);
    MockDevice::create(adapter)
}

