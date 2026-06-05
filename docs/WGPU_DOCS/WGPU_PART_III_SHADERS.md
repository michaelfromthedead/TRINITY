# WGPU_PART_III_SHADERS.md — Shader Compilation

> **TOC Reference**: Part III, Chapter 5
> **Purpose**: Complete specification of WGSL and Naga shader compilation for TRINITY
> **Generated**: 2026-05-27

---

# Chapter 5: WGSL & Naga

## 5.1 WGSL Language

### 5.1.1 WGSL Syntax Fundamentals

WGSL (WebGPU Shading Language) is the shader language for wgpu. Key characteristics:

- **C-like syntax** with Rust-inspired ownership semantics
- **Strongly typed** with no implicit conversions
- **Explicit address spaces** for memory safety
- **GPU-specific built-ins** for common operations

```wgsl
// Basic structure
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var albedo_texture: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4(in.position, 1.0);
    out.world_position = in.position;
    out.world_normal = in.normal;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = textureSample(albedo_texture, linear_sampler, in.uv);
    let n = normalize(in.world_normal);
    let light_dir = normalize(vec3(1.0, 1.0, 1.0));
    let ndotl = max(dot(n, light_dir), 0.0);
    return vec4(albedo.rgb * ndotl, albedo.a);
}
```

### 5.1.2 Types

**Scalar types:**
```wgsl
let b: bool = true;
let i: i32 = -42;
let u: u32 = 42u;
let f: f32 = 3.14;
let h: f16 = 3.14h;  // Requires f16 feature
```

**Vector types:**
```wgsl
let v2f: vec2<f32> = vec2(1.0, 2.0);
let v3f: vec3<f32> = vec3(1.0, 2.0, 3.0);
let v4f: vec4<f32> = vec4(1.0, 2.0, 3.0, 4.0);
let v4i: vec4<i32> = vec4(1, 2, 3, 4);
let v4u: vec4<u32> = vec4(1u, 2u, 3u, 4u);

// Swizzling
let xy = v4f.xy;
let rgb = v4f.rgb;
let rrr = v3f.xxx;
let wzyx = v4f.wzyx;
```

**Matrix types:**
```wgsl
let m2x2: mat2x2<f32> = mat2x2(1.0, 0.0, 0.0, 1.0);
let m3x3: mat3x3<f32> = mat3x3(/* 9 values or 3 vec3 */);
let m4x4: mat4x4<f32> = mat4x4(/* 16 values or 4 vec4 */);
let m3x4: mat3x4<f32> = mat3x4(/* 3 rows, 4 columns */);

// Matrix access
let col0: vec4<f32> = m4x4[0];
let element: f32 = m4x4[1][2];
```

**Array types:**
```wgsl
// Fixed-size array
let arr: array<f32, 4> = array<f32, 4>(1.0, 2.0, 3.0, 4.0);

// Runtime-sized array (storage buffer only)
struct LightBuffer {
    count: u32,
    lights: array<Light>,  // Runtime-sized
}
```

**Struct types:**
```wgsl
struct Light {
    position: vec3<f32>,
    _pad0: f32,
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    range: f32,
}

struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    position: vec4<f32>,
    near_far_fov_aspect: vec4<f32>,
}
```

### 5.1.3 Address Spaces

```wgsl
// Function scope (local variables)
var<function> local_var: f32 = 0.0;
fn foo() {
    var x: f32 = 1.0;  // Implicit function address space
}

// Private (per-invocation global)
var<private> per_invocation: f32 = 0.0;

// Workgroup (shared within compute workgroup)
var<workgroup> shared_data: array<f32, 256>;

// Uniform (read-only, externally provided)
@group(0) @binding(0) var<uniform> camera: CameraUniform;

// Storage (read or read-write, externally provided)
@group(0) @binding(1) var<storage, read> vertex_data: array<Vertex>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

// Handle (textures, samplers)
@group(0) @binding(3) var albedo: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
```

### 5.1.4 Built-in Functions

**Math functions:**
```wgsl
// Trigonometry
sin(x), cos(x), tan(x)
asin(x), acos(x), atan(x), atan2(y, x)
sinh(x), cosh(x), tanh(x)

// Exponential
exp(x), exp2(x)
log(x), log2(x)
pow(base, exp)
sqrt(x), inverseSqrt(x)

// Common
abs(x), sign(x)
floor(x), ceil(x), round(x), trunc(x), fract(x)
min(a, b), max(a, b)
clamp(x, low, high)
mix(a, b, t)  // Linear interpolation
step(edge, x)
smoothstep(low, high, x)

// Geometric
dot(a, b)
cross(a, b)
length(v)
distance(a, b)
normalize(v)
reflect(I, N)
refract(I, N, eta)
faceForward(N, I, Nref)
```

**Vector/matrix functions:**
```wgsl
// Vector construction
vec2(x, y)
vec3(xy, z), vec3(x, yz)
vec4(xyz, w), vec4(x, yzw), vec4(xy, zw)

// Matrix construction and operations
transpose(m)
determinant(m)

// Vector operations
all(bvec)  // All components true
any(bvec)  // Any component true
select(f, t, cond)  // Component-wise select
```

**Texture functions:**
```wgsl
// Sampling
textureSample(t, s, coord)
textureSampleBias(t, s, coord, bias)
textureSampleLevel(t, s, coord, level)
textureSampleGrad(t, s, coord, ddx, ddy)
textureSampleCompare(t, s, coord, depth_ref)
textureSampleCompareLevel(t, s, coord, depth_ref, level)

// Loading (no filtering)
textureLoad(t, coord, level)
textureStore(t, coord, value)

// Queries
textureDimensions(t)
textureDimensions(t, level)
textureNumLayers(t)
textureNumLevels(t)
textureNumSamples(t)

// Gather (sample 4 texels for bilinear)
textureGather(component, t, s, coord)
textureGatherCompare(t, s, coord, depth_ref)
```

**Derivative functions (fragment only):**
```wgsl
dpdx(v)       // Partial derivative in x
dpdy(v)       // Partial derivative in y
dpdxCoarse(v) // Coarse derivative in x
dpdyCoarse(v) // Coarse derivative in y
dpdxFine(v)   // Fine derivative in x
dpdyFine(v)   // Fine derivative in y
fwidth(v)     // abs(dpdx(v)) + abs(dpdy(v))
```

**Atomic functions (storage/workgroup):**
```wgsl
atomicLoad(ptr)
atomicStore(ptr, value)
atomicAdd(ptr, value)
atomicSub(ptr, value)
atomicMax(ptr, value)
atomicMin(ptr, value)
atomicAnd(ptr, value)
atomicOr(ptr, value)
atomicXor(ptr, value)
atomicExchange(ptr, value)
atomicCompareExchangeWeak(ptr, expected, value)
```

**Synchronization:**
```wgsl
workgroupBarrier()     // Sync workgroup execution + memory
storageBarrier()       // Sync storage buffer memory
textureBarrier()       // Sync texture memory (where supported)
```

**Pack/unpack:**
```wgsl
pack4x8snorm(v)        // vec4<f32> -> u32
pack4x8unorm(v)        // vec4<f32> -> u32
pack2x16snorm(v)       // vec2<f32> -> u32
pack2x16unorm(v)       // vec2<f32> -> u32
pack2x16float(v)       // vec2<f32> -> u32
unpack4x8snorm(u)      // u32 -> vec4<f32>
unpack4x8unorm(u)      // u32 -> vec4<f32>
unpack2x16snorm(u)     // u32 -> vec2<f32>
unpack2x16unorm(u)     // u32 -> vec2<f32>
unpack2x16float(u)     // u32 -> vec2<f32>
```

### 5.1.5 Attributes

**Entry point attributes:**
```wgsl
@vertex                  // Vertex shader entry
@fragment                // Fragment shader entry
@compute                 // Compute shader entry
@workgroup_size(x, y, z) // Compute workgroup dimensions
```

**Binding attributes:**
```wgsl
@group(n)        // Bind group index (0-3)
@binding(n)      // Binding index within group
@location(n)     // Vertex input/output location
@builtin(name)   // Built-in variable
```

**Interpolation attributes:**
```wgsl
@interpolate(perspective)        // Default
@interpolate(perspective, center) // Default
@interpolate(perspective, centroid)
@interpolate(perspective, sample)
@interpolate(linear)
@interpolate(linear, center)
@interpolate(linear, centroid)
@interpolate(linear, sample)
@interpolate(flat)               // No interpolation
```

**Other attributes:**
```wgsl
@align(n)        // Struct member alignment
@size(n)         // Struct member size
@id(n)           // Override constant ID
@must_use        // Result must be used
@diagnostic(...)  // Compiler diagnostics
```

### 5.1.6 Built-in Variables

**Vertex shader:**
```wgsl
@builtin(vertex_index) vertex_index: u32
@builtin(instance_index) instance_index: u32
@builtin(position) position: vec4<f32>  // Output
```

**Fragment shader:**
```wgsl
@builtin(position) frag_coord: vec4<f32>  // Input, window coordinates
@builtin(front_facing) front_facing: bool
@builtin(sample_index) sample_index: u32
@builtin(sample_mask) sample_mask: u32
@builtin(frag_depth) frag_depth: f32  // Output
```

**Compute shader:**
```wgsl
@builtin(global_invocation_id) global_id: vec3<u32>
@builtin(local_invocation_id) local_id: vec3<u32>
@builtin(local_invocation_index) local_index: u32
@builtin(workgroup_id) workgroup_id: vec3<u32>
@builtin(num_workgroups) num_workgroups: vec3<u32>
```

### 5.1.7 WGSL Extensions

```wgsl
// Enable extensions at top of shader
enable f16;                    // 16-bit floats
enable clip_distances;         // Clip distances
enable dual_source_blending;   // Dual source blend
enable chromium_disable_uniformity_analysis;  // Debug only

// Diagnostic control
diagnostic(off, derivative_uniformity);
diagnostic(warning, some_diagnostic);
diagnostic(error, some_diagnostic);
```

---

## 5.2 Naga Compiler Pipeline

### 5.2.1 Naga Architecture Overview

Naga is the shader compiler used by wgpu:

```
Source Code (WGSL/GLSL/SPIR-V)
         ↓
    [Frontend]
         ↓
    Naga IR (Module)
         ↓
    [Validation]
         ↓
    [Optimization] (optional)
         ↓
    [Backend]
         ↓
Target Code (SPIR-V/MSL/HLSL/GLSL/WGSL)
```

### 5.2.2 Frontend: WGSL → IR

```rust
use naga::{front::wgsl, valid::{Validator, ValidationFlags}};

let source = r#"
    @vertex
    fn main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
        return vec4(pos, 1.0);
    }
"#;

// Parse WGSL to Naga IR
let module = wgsl::parse_str(source)?;

// Validate
let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
let info = validator.validate(&module)?;
```

### 5.2.3 IR Representation

```rust
pub struct Module {
    pub types: Arena<Type>,
    pub constants: Arena<Constant>,
    pub global_variables: Arena<GlobalVariable>,
    pub functions: Arena<Function>,
    pub entry_points: Vec<EntryPoint>,
}

pub struct Function {
    pub name: Option<String>,
    pub arguments: Vec<FunctionArgument>,
    pub result: Option<FunctionResult>,
    pub local_variables: Arena<LocalVariable>,
    pub expressions: Arena<Expression>,
    pub body: Block,
}

pub enum Expression {
    Literal(Literal),
    Constant(Handle<Constant>),
    ZeroValue(Handle<Type>),
    Compose { ty: Handle<Type>, components: Vec<Handle<Expression>> },
    Access { base: Handle<Expression>, index: Handle<Expression> },
    AccessIndex { base: Handle<Expression>, index: u32 },
    Splat { size: VectorSize, value: Handle<Expression> },
    Swizzle { size: VectorSize, vector: Handle<Expression>, pattern: [SwizzleComponent; 4] },
    FunctionArgument(u32),
    GlobalVariable(Handle<GlobalVariable>),
    LocalVariable(Handle<LocalVariable>),
    Load { pointer: Handle<Expression> },
    ImageSample { ... },
    ImageLoad { ... },
    ImageQuery { ... },
    Unary { op: UnaryOperator, expr: Handle<Expression> },
    Binary { op: BinaryOperator, left: Handle<Expression>, right: Handle<Expression> },
    Select { condition: Handle<Expression>, accept: Handle<Expression>, reject: Handle<Expression> },
    Derivative { axis: DerivativeAxis, ctrl: DerivativeControl, expr: Handle<Expression> },
    Relational { fun: RelationalFunction, argument: Handle<Expression> },
    Math { fun: MathFunction, arg: Handle<Expression>, ... },
    As { expr: Handle<Expression>, kind: ScalarKind, convert: Option<u8> },
    CallResult(Handle<Function>),
    AtomicResult { ... },
    WorkGroupUniformLoadResult { ... },
    ArrayLength(Handle<Expression>),
    RayQueryProceedResult,
    RayQueryGetIntersection { ... },
}
```

### 5.2.4 Validation Passes

```rust
pub struct ValidationFlags: u32 {
    const STRUCT_LAYOUTS = 0x1;      // Check struct layouts
    const CONSTANTS = 0x2;           // Validate constants
    const BLOCK_ACTION = 0x4;        // Block termination
    const CONTROL_FLOW_UNIFORMITY = 0x8;  // Uniformity analysis
    const EXPRESSIONS = 0x10;        // Expression validation
    const BINDINGS = 0x20;           // Binding validation
}

pub struct Capabilities: u32 {
    const PUSH_CONSTANT = 0x1;
    const FLOAT64 = 0x2;
    const PRIMITIVE_INDEX = 0x4;
    const SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING = 0x8;
    const UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING = 0x10;
    const SAMPLER_NON_UNIFORM_INDEXING = 0x20;
    const CLIP_DISTANCE = 0x40;
    const CULL_DISTANCE = 0x80;
    const STORAGE_TEXTURE_16BIT_NORM_FORMATS = 0x100;
    const MULTIVIEW = 0x200;
    const EARLY_DEPTH_TEST = 0x400;
    const MULTISAMPLED_SHADING = 0x800;
    const DUAL_SOURCE_BLENDING = 0x1000;
    const CUBE_ARRAY_TEXTURES = 0x2000;
    const SUBGROUP = 0x4000;
    const SUBGROUP_BARRIER = 0x8000;
    const RAY_QUERY = 0x10000;
}
```

### 5.2.5 Backend Targets

```rust
use naga::back::{spv, msl, hlsl, glsl, wgsl};

// SPIR-V output
let spv_options = spv::Options {
    lang_version: (1, 5),
    flags: spv::WriterFlags::ADJUST_COORDINATE_SPACE,
    binding_map: Default::default(),
    capabilities: None,
    bounds_check_policies: Default::default(),
    zero_initialize_workgroup_memory: true,
    debug_info: None,
};
let spv_binary = spv::write_vec(&module, &info, &spv_options, None)?;

// MSL output (Metal)
let msl_options = msl::Options {
    lang_version: (2, 4),
    per_entry_point_map: Default::default(),
    inline_samplers: vec![],
    spirv_cross_compatibility: false,
    fake_missing_bindings: false,
    bounds_check_policies: Default::default(),
    zero_initialize_workgroup_memory: true,
};
let msl_string = msl::write_string(&module, &info, &msl_options, &pipeline_options)?;

// HLSL output
let hlsl_options = hlsl::Options {
    shader_model: hlsl::ShaderModel::V6_5,
    binding_map: Default::default(),
    fake_missing_bindings: false,
    special_constants_binding: None,
    push_constants_target: None,
    zero_initialize_workgroup_memory: true,
};
let hlsl_string = hlsl::write_string(&module, &info, &hlsl_options)?;

// GLSL output
let glsl_options = glsl::Options {
    version: glsl::Version::Desktop(450),
    writer_flags: glsl::WriterFlags::empty(),
    binding_map: Default::default(),
    zero_initialize_workgroup_memory: true,
};
let mut glsl_string = String::new();
let mut writer = glsl::Writer::new(&mut glsl_string, &module, &info, &glsl_options, &pipeline_options, Default::default())?;
writer.write()?;
```

### 5.2.6 Compilation Caching

```rust
pub struct ShaderCache {
    cache_dir: PathBuf,
    in_memory: HashMap<ShaderKey, Arc<ShaderModule>>,
}

#[derive(Hash, Eq, PartialEq)]
struct ShaderKey {
    source_hash: u64,
    defines: Vec<(String, String)>,
    target: CompileTarget,
}

impl ShaderCache {
    pub fn get_or_compile(
        &mut self,
        device: &Device,
        source: &str,
        defines: &[(&str, &str)],
    ) -> Result<Arc<ShaderModule>, ShaderError> {
        let key = ShaderKey::new(source, defines);
        
        // Check in-memory cache
        if let Some(module) = self.in_memory.get(&key) {
            return Ok(module.clone());
        }
        
        // Check disk cache
        let cache_path = self.cache_dir.join(format!("{:016x}.spv", key.hash()));
        if let Ok(bytes) = std::fs::read(&cache_path) {
            let module = unsafe { device.create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
                label: None,
                source: bytemuck::cast_slice(&bytes).into(),
            }) };
            let module = Arc::new(module);
            self.in_memory.insert(key, module.clone());
            return Ok(module);
        }
        
        // Compile
        let module = self.compile(device, source, defines)?;
        
        // Cache to disk
        // ... serialize and write
        
        let module = Arc::new(module);
        self.in_memory.insert(key, module.clone());
        Ok(module)
    }
}
```

### 5.2.7 TRINITY's Shader Hot-Reload System

```rust
pub struct ShaderHotReload {
    watcher: notify::RecommendedWatcher,
    shader_sources: HashMap<PathBuf, ShaderSource>,
    pending_reloads: Receiver<PathBuf>,
    compiled_modules: HashMap<ShaderId, Arc<ShaderModule>>,
}

impl ShaderHotReload {
    pub fn new(shader_dir: &Path) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, Error>| {
            if let Ok(event) = res {
                if event.kind.is_modify() {
                    for path in event.paths {
                        if path.extension() == Some(OsStr::new("wgsl")) {
                            let _ = tx.send(path);
                        }
                    }
                }
            }
        }).unwrap();
        
        watcher.watch(shader_dir, RecursiveMode::Recursive).unwrap();
        
        Self {
            watcher,
            shader_sources: HashMap::new(),
            pending_reloads: rx,
            compiled_modules: HashMap::new(),
        }
    }
    
    pub fn poll_reloads(&mut self, device: &Device) -> Vec<ShaderId> {
        let mut reloaded = Vec::new();
        
        while let Ok(path) = self.pending_reloads.try_recv() {
            if let Some(source) = self.shader_sources.get_mut(&path) {
                match source.reload_and_compile(device) {
                    Ok(module) => {
                        let id = source.id;
                        self.compiled_modules.insert(id, Arc::new(module));
                        reloaded.push(id);
                        log::info!("Hot-reloaded shader: {:?}", path);
                    }
                    Err(e) => {
                        log::error!("Shader reload failed: {:?}: {}", path, e);
                    }
                }
            }
        }
        
        reloaded
    }
}
```

---

## 5.3 Shader Modules

### 5.3.1 Shader Module Creation

```rust
// From WGSL source
let module = device.create_shader_module(ShaderModuleDescriptor {
    label: Some("My Shader"),
    source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
});

// From pre-compiled SPIR-V (unsafe, skips validation)
let module = unsafe {
    device.create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
        label: Some("SPIR-V Shader"),
        source: spirv_binary.into(),
    })
};
```

### 5.3.2 Compilation Error Handling

```rust
pub fn compile_shader(device: &Device, source: &str) -> Result<ShaderModule, ShaderError> {
    // Pre-validate with Naga for better error messages
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(m) => m,
        Err(e) => {
            return Err(ShaderError::ParseError {
                message: e.emit_to_string(source),
                location: e.location(source).map(|l| ShaderLocation {
                    line: l.line_number,
                    column: l.line_position,
                }),
            });
        }
    };
    
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    
    if let Err(e) = validator.validate(&module) {
        return Err(ShaderError::ValidationError {
            message: format!("{:?}", e),
        });
    }
    
    // Create wgpu module
    Ok(device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(source.into()),
    }))
}
```

### 5.3.3 Shader Reflection

```rust
pub struct ShaderReflection {
    pub entry_points: Vec<EntryPointInfo>,
    pub bindings: Vec<BindingInfo>,
    pub push_constants: Option<PushConstantInfo>,
}

pub struct EntryPointInfo {
    pub name: String,
    pub stage: ShaderStage,
    pub workgroup_size: Option<[u32; 3]>,
}

pub struct BindingInfo {
    pub group: u32,
    pub binding: u32,
    pub name: Option<String>,
    pub ty: BindingType,
    pub count: Option<u32>,
}

impl ShaderReflection {
    pub fn from_naga_module(module: &naga::Module, info: &naga::valid::ModuleInfo) -> Self {
        let mut bindings = Vec::new();
        
        for (handle, var) in module.global_variables.iter() {
            if let Some(binding) = &var.binding {
                bindings.push(BindingInfo {
                    group: binding.group,
                    binding: binding.binding,
                    name: var.name.clone(),
                    ty: convert_binding_type(&module.types[var.ty]),
                    count: None, // Array count
                });
            }
        }
        
        let entry_points = module.entry_points.iter().map(|ep| {
            EntryPointInfo {
                name: ep.name.clone(),
                stage: convert_stage(ep.stage),
                workgroup_size: ep.workgroup_size,
            }
        }).collect();
        
        Self {
            entry_points,
            bindings,
            push_constants: None, // TODO
        }
    }
}
```

### 5.3.4 Shader Module Caching

See 5.2.6 above.

### 5.3.5 Runtime vs Ahead-of-Time Compilation

**Runtime (development):**
```rust
pub fn load_shader_runtime(device: &Device, path: &Path) -> Result<ShaderModule, Error> {
    let source = std::fs::read_to_string(path)?;
    Ok(device.create_shader_module(ShaderModuleDescriptor {
        label: Some(path.to_str().unwrap()),
        source: ShaderSource::Wgsl(source.into()),
    }))
}
```

**Ahead-of-time (release):**
```rust
// Build script compiles to SPIR-V
// build.rs
fn main() {
    let shader_dir = Path::new("shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    
    for entry in std::fs::read_dir(shader_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension() == Some(OsStr::new("wgsl")) {
            let source = std::fs::read_to_string(&path).unwrap();
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            ).validate(&module).unwrap();
            
            let spv = naga::back::spv::write_vec(&module, &info, &Default::default(), None).unwrap();
            
            let out_path = Path::new(&out_dir).join(path.file_stem().unwrap()).with_extension("spv");
            std::fs::write(&out_path, bytemuck::cast_slice(&spv)).unwrap();
        }
    }
}

// Runtime loading
static SHADER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shader.spv"));

fn load_shader_aot(device: &Device) -> ShaderModule {
    unsafe {
        device.create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
            label: Some("AOT Shader"),
            source: bytemuck::cast_slice(SHADER_SPV).into(),
        })
    }
}
```

---

## 5.4 Shader Specialization

### 5.4.1 Override Constants

```wgsl
// Shader with override constants
@id(0) override WORKGROUP_SIZE_X: u32 = 8;
@id(1) override WORKGROUP_SIZE_Y: u32 = 8;
@id(2) override USE_NORMAL_MAP: bool = true;
@id(3) override MAX_LIGHTS: u32 = 16;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main() {
    if (USE_NORMAL_MAP) {
        // ...
    }
}
```

**Setting overrides at pipeline creation:**
```rust
let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    label: Some("Compute Pipeline"),
    layout: Some(&layout),
    module: &shader_module,
    entry_point: "main",
    compilation_options: PipelineCompilationOptions {
        constants: &[
            ("WORKGROUP_SIZE_X", 16.0),  // Override to 16
            ("WORKGROUP_SIZE_Y", 16.0),
            ("USE_NORMAL_MAP", 1.0),     // true = 1.0, false = 0.0
            ("MAX_LIGHTS", 32.0),
        ].into_iter().collect(),
        zero_initialize_workgroup_memory: true,
    },
});
```

### 5.4.2 Specialization Constant Patterns

```rust
pub struct SpecializationConstants {
    pub workgroup_size: [u32; 3],
    pub use_normal_maps: bool,
    pub use_emissive: bool,
    pub max_lights: u32,
    pub quality_level: u32,
}

impl SpecializationConstants {
    pub fn to_wgpu_constants(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("WORKGROUP_SIZE_X".into(), self.workgroup_size[0] as f64);
        map.insert("WORKGROUP_SIZE_Y".into(), self.workgroup_size[1] as f64);
        map.insert("WORKGROUP_SIZE_Z".into(), self.workgroup_size[2] as f64);
        map.insert("USE_NORMAL_MAPS".into(), if self.use_normal_maps { 1.0 } else { 0.0 });
        map.insert("USE_EMISSIVE".into(), if self.use_emissive { 1.0 } else { 0.0 });
        map.insert("MAX_LIGHTS".into(), self.max_lights as f64);
        map.insert("QUALITY_LEVEL".into(), self.quality_level as f64);
        map
    }
}
```

### 5.4.3 Shader Permutation Management

```rust
pub struct ShaderPermutationManager {
    base_source: String,
    permutations: HashMap<PermutationKey, Arc<ShaderModule>>,
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct PermutationKey {
    pub features: FeatureFlags,
    pub quality: QualityLevel,
}

bitflags! {
    pub struct FeatureFlags: u32 {
        const NORMAL_MAPS = 1 << 0;
        const EMISSIVE = 1 << 1;
        const ALPHA_TEST = 1 << 2;
        const SKINNING = 1 << 3;
        const INSTANCING = 1 << 4;
        const SHADOWS = 1 << 5;
        const FOG = 1 << 6;
    }
}

impl ShaderPermutationManager {
    pub fn get_or_create(
        &mut self,
        device: &Device,
        key: &PermutationKey,
    ) -> Arc<ShaderModule> {
        if let Some(module) = self.permutations.get(key) {
            return module.clone();
        }
        
        // Generate preprocessed source
        let source = self.preprocess(&self.base_source, key);
        
        let module = Arc::new(device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("Permutation {:?}", key)),
            source: ShaderSource::Wgsl(source.into()),
        }));
        
        self.permutations.insert(key.clone(), module.clone());
        module
    }
    
    fn preprocess(&self, source: &str, key: &PermutationKey) -> String {
        // Simple preprocessor implementation
        let mut result = String::new();
        
        // Add defines based on features
        if key.features.contains(FeatureFlags::NORMAL_MAPS) {
            result.push_str("const USE_NORMAL_MAPS: bool = true;\n");
        } else {
            result.push_str("const USE_NORMAL_MAPS: bool = false;\n");
        }
        // ... other features
        
        result.push_str(source);
        result
    }
}
```

### 5.4.4 TRINITY's Shader Variant System

```rust
pub struct ShaderVariantSystem {
    registry: HashMap<ShaderAssetId, ShaderAsset>,
    variants: HashMap<VariantKey, CompiledVariant>,
    compiler: ShaderCompiler,
}

pub struct ShaderAsset {
    pub id: ShaderAssetId,
    pub name: String,
    pub source: String,
    pub features: Vec<ShaderFeature>,
    pub specialization_constants: Vec<SpecConstant>,
}

pub struct ShaderFeature {
    pub name: String,
    pub define: String,
    pub default: bool,
}

pub struct SpecConstant {
    pub id: u32,
    pub name: String,
    pub ty: SpecConstType,
    pub default: f64,
}

impl ShaderVariantSystem {
    pub fn get_variant(
        &mut self,
        device: &Device,
        shader_id: ShaderAssetId,
        features: FeatureFlags,
        spec_constants: &HashMap<String, f64>,
    ) -> Result<&CompiledVariant, ShaderError> {
        let key = VariantKey { shader_id, features, spec_hash: hash_spec_constants(spec_constants) };
        
        if !self.variants.contains_key(&key) {
            let asset = self.registry.get(&shader_id).ok_or(ShaderError::NotFound)?;
            let variant = self.compiler.compile_variant(device, asset, features, spec_constants)?;
            self.variants.insert(key.clone(), variant);
        }
        
        Ok(self.variants.get(&key).unwrap())
    }
    
    pub fn precompile_common_variants(&mut self, device: &Device) {
        // Precompile frequently used combinations
        let common_features = [
            FeatureFlags::empty(),
            FeatureFlags::NORMAL_MAPS,
            FeatureFlags::NORMAL_MAPS | FeatureFlags::SHADOWS,
            FeatureFlags::NORMAL_MAPS | FeatureFlags::SHADOWS | FeatureFlags::FOG,
            FeatureFlags::all(),
        ];
        
        for (id, asset) in &self.registry {
            for features in &common_features {
                let _ = self.get_variant(device, *id, *features, &HashMap::new());
            }
        }
    }
}
```

---

# TRINITY Shader Module Architecture

```
crates/renderer-backend/src/shader/
├── mod.rs              # Module root
├── compiler.rs         # Naga integration, compilation
├── cache.rs            # Disk and memory caching
├── hot_reload.rs       # Development hot-reload
├── reflection.rs       # Shader reflection
├── permutation.rs      # Permutation management
├── variant.rs          # Variant system
├── preprocessor.rs     # Simple preprocessor
└── library.rs          # Common shader includes
```

```
shaders/
├── common/
│   ├── math.wgsl       # Math utilities
│   ├── packing.wgsl    # Pack/unpack functions
│   ├── sampling.wgsl   # Texture sampling utilities
│   ├── pbr.wgsl        # PBR functions
│   ├── lighting.wgsl   # Lighting calculations
│   └── tonemapping.wgsl # Tonemapping operators
├── vertex/
│   ├── static_mesh.wgsl
│   ├── skinned_mesh.wgsl
│   └── instanced.wgsl
├── fragment/
│   ├── gbuffer.wgsl
│   ├── forward.wgsl
│   └── deferred_lighting.wgsl
├── compute/
│   ├── culling.wgsl
│   ├── histogram.wgsl
│   ├── blur.wgsl
│   └── mip_generation.wgsl
└── rt/
    ├── shadow_query.wgsl
    ├── reflection.wgsl
    └── gi.wgsl
```

---

*End of WGPU_PART_III_SHADERS.md*
