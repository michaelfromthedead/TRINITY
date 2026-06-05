//! RHI pipeline mapping layer.
//!
//! Provides RHI-level pipeline descriptor types and helpers that sit between
//! the [`pipeline`](crate::pipeline) module (cached pipeline table) and the
//! raw wgpu API.
//!
//! Owned descriptor structs (`VertexAttribute`, `VertexBufferLayout`,
//! `BlendState`, `DepthStencilState`) let callers describe GPU pipeline state
//! without managing wgpu borrow lifetimes, while `RhiRenderPipeline`,
//! `RhiComputePipeline`, and `RhiShaderModule` wrap their wgpu counterparts
//! together with layout and source-hash metadata.
//!
//! The `create_render_pipeline` and `create_compute_pipeline` helpers provide
//! single-call construction from WGSL source strings.

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// SHA-256 helper
// ---------------------------------------------------------------------------

/// Compute the SHA-256 hash of `data` and return it as a `[u8; 32]`.
fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

// ---------------------------------------------------------------------------
// WGSL Validation (pre-validate before wgpu to avoid abort on invalid WGSL)
// ---------------------------------------------------------------------------

/// Validate WGSL source using naga. Returns Ok(()) if valid, Err(message) if invalid.
/// This prevents wgpu from aborting the process on invalid shaders.
fn validate_wgsl(source: &str) -> Result<(), String> {
    naga::front::wgsl::parse_str(source)
        .map(|_| ())
        .map_err(|e| format!("WGSL parse error: {e}"))
}

// ---------------------------------------------------------------------------
// VertexAttribute
// ---------------------------------------------------------------------------

/// Describes a single vertex attribute within a vertex buffer layout.
///
/// This is the RHI-owned counterpart of [`wgpu::VertexAttribute`], storing
/// owned data so callers do not need to manage slice lifetimes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexAttribute {
    /// The GPU data format of the attribute (e.g. `Float32x3`).
    pub format: wgpu::VertexFormat,
    /// Byte offset from the start of the vertex buffer element.
    pub offset: wgpu::BufferAddress,
    /// The `@location(...)` in the vertex shader.
    pub shader_location: u32,
}

impl VertexAttribute {
    pub fn new(
        format: wgpu::VertexFormat,
        offset: wgpu::BufferAddress,
        shader_location: u32,
    ) -> Self {
        Self {
            format,
            offset,
            shader_location,
        }
    }
}

impl From<VertexAttribute> for wgpu::VertexAttribute {
    fn from(attr: VertexAttribute) -> Self {
        wgpu::VertexAttribute {
            format: attr.format,
            offset: attr.offset,
            shader_location: attr.shader_location,
        }
    }
}

// ---------------------------------------------------------------------------
// VertexBufferLayout
// ---------------------------------------------------------------------------

/// Describes the layout of a single vertex buffer.
///
/// This is the RHI-owned counterpart of [`wgpu::VertexBufferLayout`], storing
/// an owned [`Vec`] of [`VertexAttribute`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct VertexBufferLayout {
    /// Byte stride between consecutive vertex elements.
    pub stride: wgpu::BufferAddress,
    /// Whether the buffer advances per vertex or per instance.
    pub step_mode: wgpu::VertexStepMode,
    /// The vertex attributes in this buffer.
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    pub fn new(
        stride: wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode,
        attributes: Vec<VertexAttribute>,
    ) -> Self {
        Self {
            stride,
            step_mode,
            attributes,
        }
    }
}

// ---------------------------------------------------------------------------
// BlendState
// ---------------------------------------------------------------------------

/// Describes per-target colour blending.
///
/// The same factor and operation are applied to both the colour and alpha
/// channels. When `None` is passed to [`create_render_pipeline`], wgpu's
/// default `REPLACE` behaviour is used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendState {
    /// Source blend factor (multiplies the fragment shader output).
    pub src_factor: wgpu::BlendFactor,
    /// Destination blend factor (multiplies the existing framebuffer value).
    pub dst_factor: wgpu::BlendFactor,
    /// How the source and destination are combined.
    pub blend_op: wgpu::BlendOperation,
}

impl BlendState {
    pub fn new(
        src_factor: wgpu::BlendFactor,
        dst_factor: wgpu::BlendFactor,
        blend_op: wgpu::BlendOperation,
    ) -> Self {
        Self {
            src_factor,
            dst_factor,
            blend_op,
        }
    }

    /// Convert to a [`wgpu::BlendState`] applying the same factor and
    /// operation to both colour and alpha components.
    pub fn to_wgpu(&self) -> wgpu::BlendState {
        let component = wgpu::BlendComponent {
            src_factor: self.src_factor,
            dst_factor: self.dst_factor,
            operation: self.blend_op,
        };
        wgpu::BlendState {
            color: component,
            alpha: component,
        }
    }
}

// ---------------------------------------------------------------------------
// DepthStencilState
// ---------------------------------------------------------------------------

/// Describes depth and stencil testing for a render pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthStencilState {
    /// The format of the depth/stencil attachment (e.g. `Depth32Float`).
    pub format: wgpu::TextureFormat,
    /// Whether depth values are written to the attachment.
    pub depth_write: bool,
    /// The comparison function used for depth testing.
    pub depth_compare: wgpu::CompareFunction,
    /// The stencil state (front and back face).
    pub stencil: wgpu::StencilState,
}

impl DepthStencilState {
    pub fn new(
        format: wgpu::TextureFormat,
        depth_write: bool,
        depth_compare: wgpu::CompareFunction,
        stencil: wgpu::StencilState,
    ) -> Self {
        Self {
            format,
            depth_write,
            depth_compare,
            stencil,
        }
    }

    /// Convert to a [`wgpu::DepthStencilState`] with default bias.
    pub fn to_wgpu(&self) -> wgpu::DepthStencilState {
        wgpu::DepthStencilState {
            format: self.format,
            depth_write_enabled: self.depth_write,
            depth_compare: self.depth_compare,
            stencil: self.stencil.clone(),
            bias: wgpu::DepthBiasState::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// RhiShaderModule
// ---------------------------------------------------------------------------

/// A compiled WGSL shader module together with the SHA-256 hash of its
/// source.
///
/// The source hash enables deduplication and change detection at the RHI
/// layer, complementing the [`ShaderCache`](crate::pipeline::ShaderCache)
/// which operates at the pipeline-table level.
#[derive(Debug)]
pub struct RhiShaderModule {
    /// The underlying wgpu shader module.
    pub inner: wgpu::ShaderModule,
    /// SHA-256 hash of the WGSL source that produced this module.
    pub source_hash: [u8; 32],
}

impl RhiShaderModule {
    /// Compile a WGSL source string into a shader module and record its hash.
    pub fn new(device: &wgpu::Device, source: &str) -> Self {
        let hash = sha256(source.as_bytes());
        let inner = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RhiShaderModule"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        Self { inner, source_hash: hash }
    }

    /// Borrow the underlying [`wgpu::ShaderModule`].
    pub fn inner(&self) -> &wgpu::ShaderModule {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// PipelineLayout
// ---------------------------------------------------------------------------

/// A compiled pipeline layout backed by one or more bind group layouts.
///
/// Owns its bind group layouts so callers do not need to keep them alive
/// separately.
#[derive(Debug)]
pub struct PipelineLayout {
    inner: wgpu::PipelineLayout,
    /// The bind group layouts that compose this pipeline layout.
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

impl PipelineLayout {
    /// Create a new pipeline layout from a set of bind group layouts.
    ///
    /// The layouts are provided in order of their `@group(...)` indices.
    pub fn new(
        device: &wgpu::Device,
        bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    ) -> Self {
        let refs: Vec<&wgpu::BindGroupLayout> = bind_group_layouts.iter().collect();
        let inner = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RHI PipelineLayout"),
            bind_group_layouts: &refs,
            push_constant_ranges: &[],
        });
        Self {
            inner,
            bind_group_layouts,
        }
    }

    /// Borrow the underlying [`wgpu::PipelineLayout`].
    pub fn inner(&self) -> &wgpu::PipelineLayout {
        &self.inner
    }

    /// Create bind group layouts from descriptor entries, then assemble
    /// them into a pipeline layout.
    ///
    /// Each entry slice corresponds to one `@group(N)` index in the shader.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let layout = PipelineLayout::from_entries(&device, &[
    ///     &[wgpu::BindGroupLayoutEntry {
    ///         binding: 0,
    ///         visibility: wgpu::ShaderStages::VERTEX,
    ///         ty: wgpu::BindingType::Buffer {
    ///             ty: wgpu::BufferBindingType::Uniform,
    ///             has_dynamic_offset: false,
    ///             min_binding_size: None,
    ///         },
    ///         count: None,
    ///     }],
    /// ]);
    /// ```
    pub fn from_entries(
        device: &wgpu::Device,
        groups: &[&[wgpu::BindGroupLayoutEntry]],
    ) -> Self {
        let layouts: Vec<wgpu::BindGroupLayout> = groups
            .iter()
            .enumerate()
            .map(|(i, entries)| {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("RHI BGL group {}", i)),
                    entries,
                })
            })
            .collect();
        Self::new(device, layouts)
    }
}

// ---------------------------------------------------------------------------
// RhiRenderPipeline
// ---------------------------------------------------------------------------

/// A compiled render pipeline together with its layout information.
///
/// Created via [`create_render_pipeline`] or by manually wrapping a
/// [`wgpu::RenderPipeline`].
pub struct RhiRenderPipeline {
    /// The underlying wgpu render pipeline.
    pub inner: wgpu::RenderPipeline,
    /// Whether a custom pipeline layout was provided at creation time.
    pub has_custom_layout: bool,
}

impl RhiRenderPipeline {
    /// Borrow the underlying [`wgpu::RenderPipeline`].
    pub fn inner(&self) -> &wgpu::RenderPipeline {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// RhiComputePipeline
// ---------------------------------------------------------------------------

/// A compiled compute pipeline together with its layout information.
///
/// Created via [`create_compute_pipeline`] or by manually wrapping a
/// [`wgpu::ComputePipeline`].
pub struct RhiComputePipeline {
    /// The underlying wgpu compute pipeline.
    pub inner: wgpu::ComputePipeline,
    /// Whether a custom pipeline layout was provided at creation time.
    pub has_custom_layout: bool,
}

impl RhiComputePipeline {
    /// Borrow the underlying [`wgpu::ComputePipeline`].
    pub fn inner(&self) -> &wgpu::ComputePipeline {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// create_render_pipeline
// ---------------------------------------------------------------------------

/// Compile a full render pipeline from WGSL source strings.
///
/// Uses `vs_main` and `fs_main` as the default vertex / fragment entry-point
/// function names.  For pipelines with non-standard entry points callers
/// should construct the pipeline directly via `wgpu::Device`.
///
/// # Arguments
///
/// * `device`         — The wgpu device.
/// * `layout`         — Optional [`PipelineLayout`]. If `None`, wgpu derives
///                      a default layout from shader reflection.
/// * `vs_src`         — WGSL vertex shader source.
/// * `fs_src`         — WGSL fragment shader source.
/// * `vertex_layouts` — Vertex buffer layouts describing input vertex data.
/// * `blend`          — Optional blend state; `None` uses wgpu's `REPLACE`.
/// * `depth`          — Optional depth/stencil state; `None` disables depth.
/// * `color_format`   — The format of the colour render target.
///
/// Returns `Err(msg)` if wgpu panics during shader or pipeline creation.
pub fn create_render_pipeline(
    device: &wgpu::Device,
    layout: Option<&PipelineLayout>,
    vs_src: &str,
    fs_src: &str,
    vertex_layouts: &[VertexBufferLayout],
    blend: Option<BlendState>,
    depth: Option<DepthStencilState>,
    color_format: wgpu::TextureFormat,
) -> Result<RhiRenderPipeline, String> {
    // -- Validate WGSL before passing to wgpu (prevents abort on invalid WGSL) --
    validate_wgsl(vs_src).map_err(|e| format!("Vertex shader: {e}"))?;
    validate_wgsl(fs_src).map_err(|e| format!("Fragment shader: {e}"))?;

    // -- Shader modules ----------------------------------------------------
    let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("RHI render vs"),
        source: wgpu::ShaderSource::Wgsl(vs_src.into()),
    });
    let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("RHI render fs"),
        source: wgpu::ShaderSource::Wgsl(fs_src.into()),
    });

    // -- Convert vertex layouts to wgpu borrows ----------------------------
    // Keep the converted attribute vectors alive so the wgpu layouts can
    // reference them during pipeline creation.
    let wgpu_attr_vecs: Vec<Vec<wgpu::VertexAttribute>> = vertex_layouts
        .iter()
        .map(|vl| {
            vl.attributes
                .iter()
                .map(|a| wgpu::VertexAttribute {
                    format: a.format,
                    offset: a.offset,
                    shader_location: a.shader_location,
                })
                .collect()
        })
        .collect();

    let wgpu_layouts: Vec<wgpu::VertexBufferLayout<'_>> = vertex_layouts
        .iter()
        .enumerate()
        .map(|(i, vl)| wgpu::VertexBufferLayout {
            array_stride: vl.stride,
            step_mode: vl.step_mode,
            attributes: &wgpu_attr_vecs[i],
        })
        .collect();

    // -- Convert blend state -----------------------------------------------
    let blend_state = blend.map(|b| {
        let component = wgpu::BlendComponent {
            src_factor: b.src_factor,
            dst_factor: b.dst_factor,
            operation: b.blend_op,
        };
        wgpu::BlendState {
            color: component,
            alpha: component,
        }
    });

    // -- Convert depth / stencil -------------------------------------------
    let depth_state = depth.map(|d| wgpu::DepthStencilState {
        format: d.format,
        depth_write_enabled: d.depth_write,
        depth_compare: d.depth_compare,
        stencil: d.stencil.clone(),
        bias: wgpu::DepthBiasState::default(),
    });

    // -- Resolve pipeline layout reference ---------------------------------
    let pipeline_layout_ref = layout.map(PipelineLayout::inner);

    // -- Create the pipeline (catch_unwind for safety) ---------------------
    let rp = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RHI RenderPipeline"),
            layout: pipeline_layout_ref,
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: "vs_main",
                buffers: &wgpu_layouts,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: blend_state,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: depth_state,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }))
    .map_err(|panic_payload| {
        let msg = panic_payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("unknown wgpu panic");
        format!("render pipeline creation panicked: {msg}")
    })?;

    Ok(RhiRenderPipeline {
        inner: rp,
        has_custom_layout: layout.is_some(),
    })
}

// ---------------------------------------------------------------------------
// create_compute_pipeline
// ---------------------------------------------------------------------------

/// Compile a compute pipeline from a WGSL compute shader source string.
///
/// # Arguments
///
/// * `device`      — The wgpu device.
/// * `layout`      — Optional [`PipelineLayout`]. If `None`, wgpu derives a
///                    default layout from shader reflection.
/// * `cs_src`      — WGSL compute shader source.
/// * `entry_point` — Name of the `@compute` entry-point function.
///
/// Returns `Err(msg)` if wgpu panics during shader or pipeline creation.
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: Option<&PipelineLayout>,
    cs_src: &str,
    entry_point: &str,
) -> Result<RhiComputePipeline, String> {
    // -- Validate WGSL before passing to wgpu (prevents abort on invalid WGSL) --
    validate_wgsl(cs_src).map_err(|e| format!("Compute shader: {e}"))?;

    // -- Shader module -----------------------------------------------------
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("RHI compute cs"),
        source: wgpu::ShaderSource::Wgsl(cs_src.into()),
    });

    // -- Resolve pipeline layout reference ---------------------------------
    let pipeline_layout_ref = layout.map(PipelineLayout::inner);

    // -- Create the pipeline (catch_unwind for safety) ---------------------
    let cp = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RHI ComputePipeline"),
            layout: pipeline_layout_ref,
            module: &cs_module,
            entry_point,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }))
    .map_err(|panic_payload| {
        let msg = panic_payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic_payload.downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("unknown wgpu panic");
        format!("compute pipeline creation panicked: {msg}")
    })?;

    Ok(RhiComputePipeline {
        inner: cp,
        has_custom_layout: layout.is_some(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SHA-256 helper ───────────────────────────────────────────────────

    #[test]
    fn test_sha256_same_input_same_hash() {
        let a = sha256(b"vertex data");
        let b = sha256(b"vertex data");
        assert_eq!(a, b);
    }

    #[test]
    fn test_sha256_different_input_different_hash() {
        let a = sha256(b"vs_main");
        let b = sha256(b"fs_main");
        assert_ne!(a, b);
    }

    // ── VertexAttribute ──────────────────────────────────────────────────

    #[test]
    fn test_vertex_attribute_new() {
        let attr = VertexAttribute::new(
            wgpu::VertexFormat::Float32x3,
            0,
            0,
        );
        assert_eq!(attr.format, wgpu::VertexFormat::Float32x3);
        assert_eq!(attr.offset, 0);
        assert_eq!(attr.shader_location, 0);
    }

    #[test]
    fn test_vertex_attribute_into_wgpu() {
        let attr = VertexAttribute::new(
            wgpu::VertexFormat::Float32x2,
            8,
            1,
        );
        let wgpu_attr: wgpu::VertexAttribute = attr.into();
        assert_eq!(wgpu_attr.format, wgpu::VertexFormat::Float32x2);
        assert_eq!(wgpu_attr.offset, 8);
        assert_eq!(wgpu_attr.shader_location, 1);
    }

    #[test]
    fn test_vertex_attribute_debug_clone_copy() {
        let a = VertexAttribute::new(wgpu::VertexFormat::Float32, 0, 0);
        let b = a; // Copy
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(a, c);
    }

    // ── VertexBufferLayout ───────────────────────────────────────────────

    #[test]
    fn test_vertex_buffer_layout_new() {
        let attrs = vec![
            VertexAttribute::new(wgpu::VertexFormat::Float32x3, 0, 0),
            VertexAttribute::new(wgpu::VertexFormat::Float32x3, 12, 1),
        ];
        let layout = VertexBufferLayout::new(24, wgpu::VertexStepMode::Vertex, attrs);
        assert_eq!(layout.stride, 24);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
        assert_eq!(layout.attributes.len(), 2);
    }

    // ── BlendState ───────────────────────────────────────────────────────

    #[test]
    fn test_blend_state_new() {
        let bs = BlendState::new(
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::OneMinusSrcAlpha,
            wgpu::BlendOperation::Add,
        );
        assert_eq!(bs.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(bs.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(bs.blend_op, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_blend_state_to_wgpu() {
        let bs = BlendState::new(
            wgpu::BlendFactor::One,
            wgpu::BlendFactor::Zero,
            wgpu::BlendOperation::Add,
        );
        let wgpu_bs = bs.to_wgpu();
        assert_eq!(wgpu_bs.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(wgpu_bs.color.dst_factor, wgpu::BlendFactor::Zero);
        assert_eq!(wgpu_bs.alpha.src_factor, wgpu::BlendFactor::One);
        // Colour and alpha share the same component.
        assert_eq!(wgpu_bs.color, wgpu_bs.alpha);
    }

    // ── DepthStencilState ────────────────────────────────────────────────

    #[test]
    fn test_depth_stencil_state_new() {
        let stencil = wgpu::StencilState {
            front: wgpu::StencilFaceState::default(),
            back: wgpu::StencilFaceState::default(),
            read_mask: 0xFF,
            write_mask: 0xFF,
        };
        let ds = DepthStencilState::new(
            wgpu::TextureFormat::Depth32Float,
            true,
            wgpu::CompareFunction::Less,
            stencil,
        );
        assert_eq!(ds.format, wgpu::TextureFormat::Depth32Float);
        assert!(ds.depth_write);
        assert_eq!(ds.depth_compare, wgpu::CompareFunction::Less);
    }

    #[test]
    fn test_depth_stencil_state_to_wgpu() {
        let stencil = wgpu::StencilState {
            front: wgpu::StencilFaceState::default(),
            back: wgpu::StencilFaceState::default(),
            read_mask: 0x00,
            write_mask: 0xFF,
        };
        let ds = DepthStencilState::new(
            wgpu::TextureFormat::Depth32Float,
            true,
            wgpu::CompareFunction::LessEqual,
            stencil,
        );
        let wgpu_ds = ds.to_wgpu();
        assert_eq!(wgpu_ds.format, wgpu::TextureFormat::Depth32Float);
        assert!(wgpu_ds.depth_write_enabled);
        assert_eq!(wgpu_ds.depth_compare, wgpu::CompareFunction::LessEqual);
    }

    // ── RhiShaderModule (requires GPU) ───────────────────────────────────

    /// Helper: obtain a (device, queue) pair, skipping the test if no GPU
    /// is available.
    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))?;
        Some(
            pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ))
            .expect("device creation"),
        )
    }

    #[test]
    fn test_rhi_shader_module_compile() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;

        let module = RhiShaderModule::new(&device, src);
        // Source hash should be deterministic.
        assert_eq!(module.source_hash, sha256(src.as_bytes()));
    }

    #[test]
    fn test_rhi_shader_module_hash_different() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let src_a = RhiShaderModule::new(&device, "fn a() {}");
        let src_b = RhiShaderModule::new(&device, "fn b() {}");
        assert_ne!(src_a.source_hash, src_b.source_hash);
    }

    // ── PipelineLayout ───────────────────────────────────────────────────

    #[test]
    fn test_pipeline_layout_new() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test BGL"),
            entries: &[],
        });

        let layout = PipelineLayout::new(&device, vec![bgl]);
        // Pipeline layout has the expected number of bind group layouts.
        assert_eq!(layout.bind_group_layouts.len(), 1);
    }

    #[test]
    fn test_pipeline_layout_from_entries() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = PipelineLayout::from_entries(&device, &[&[entry]]);
        assert_eq!(layout.bind_group_layouts.len(), 1);
    }

    // ── create_render_pipeline ───────────────────────────────────────────

    #[test]
    fn test_create_render_pipeline_simple() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let vs_src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;
        let fs_src = r#"
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let pipeline = create_render_pipeline(
            &device,
            None,
            vs_src,
            fs_src,
            &[],
            None,
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        assert!(
            pipeline.is_ok(),
            "render pipeline should compile: {:?}",
            pipeline.err()
        );
        let rp = pipeline.unwrap();
        // No layout was provided.
        assert!(!rp.has_custom_layout);
    }

    // Test disabled: wgpu 22 aborts on invalid WGSL instead of panicking,
    // so catch_unwind cannot catch the error.
    #[test]
    fn test_create_render_pipeline_invalid_wgsl() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let bad_src = "this is not valid wgsl";
        let pipeline = create_render_pipeline(
            &device,
            None,
            bad_src,
            bad_src,
            &[],
            None,
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        // wgpu should panic and we catch it -> Err.
        assert!(pipeline.is_err(), "invalid WGSL should return Err");
    }

    #[test]
    fn test_create_render_pipeline_with_layout() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test BGL"),
            entries: &[],
        });
        let layout = PipelineLayout::new(&device, vec![bgl]);

        let vs_src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;
        let fs_src = r#"
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let pipeline = create_render_pipeline(
            &device,
            Some(&layout),
            vs_src,
            fs_src,
            &[],
            None,
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        assert!(pipeline.is_ok());
        let rp = pipeline.unwrap();
        // Layout was provided.
        assert!(rp.has_custom_layout);
    }

    #[test]
    fn test_create_render_pipeline_with_vertex_layout() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let vertex_layouts = vec![VertexBufferLayout::new(
            12,
            wgpu::VertexStepMode::Vertex,
            vec![
                VertexAttribute::new(wgpu::VertexFormat::Float32x3, 0, 0),
            ],
        )];

        let vs_src = r#"
            struct VertexInput {
                @location(0) position: vec3<f32>,
            }
            @vertex fn vs_main(v: VertexInput) -> @builtin(position) vec4<f32> {
                return vec4<f32>(v.position, 1.0);
            }
        "#;
        let fs_src = r#"
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let pipeline = create_render_pipeline(
            &device,
            None,
            vs_src,
            fs_src,
            &vertex_layouts,
            None,
            None,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        assert!(pipeline.is_ok());
    }

    // ── create_compute_pipeline ──────────────────────────────────────────

    #[test]
    fn test_create_compute_pipeline_simple() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let cs_src = r#"
            @group(0) @binding(0) var<storage, read_write> buf: array<f32>;

            @compute @workgroup_size(64)
            fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
                buf[id.x] = buf[id.x] * 2.0;
            }
        "#;

        let pipeline = create_compute_pipeline(
            &device,
            None,
            cs_src,
            "cs_main",
        );

        assert!(
            pipeline.is_ok(),
            "compute pipeline should compile: {:?}",
            pipeline.err()
        );
        let cp = pipeline.unwrap();
        assert!(!cp.has_custom_layout);
    }

    // Test disabled: wgpu 22 aborts on invalid WGSL instead of panicking,
    // so catch_unwind cannot catch the error.
    #[test]
    fn test_create_compute_pipeline_invalid_wgsl() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let pipeline = create_compute_pipeline(
            &device,
            None,
            "not valid wgsl at all",
            "main",
        );

        assert!(pipeline.is_err(), "invalid WGSL should return Err");
    }

    #[test]
    fn test_create_compute_pipeline_with_layout() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        // Storage buffer bind group layout.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = PipelineLayout::new(&device, vec![bgl]);

        let cs_src = r#"
            @group(0) @binding(0) var<storage, read_write> buf: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                buf[id.x] = buf[id.x] * 2.0;
            }
        "#;

        let pipeline = create_compute_pipeline(
            &device,
            Some(&layout),
            cs_src,
            "main",
        );

        assert!(pipeline.is_ok());
        let cp = pipeline.unwrap();
        assert!(cp.has_custom_layout);
    }

    // ── Integration: render pipeline + depth/blend ───────────────────────

    #[test]
    fn test_create_render_pipeline_with_depth_and_blend() {
        let (device, _queue) = match create_test_device() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: no GPU adapter available");
                return;
            }
        };

        let vs_src = r#"
            @vertex fn vs_main() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;
        let fs_src = r#"
            @fragment fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
        "#;

        let blend = BlendState::new(
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::OneMinusSrcAlpha,
            wgpu::BlendOperation::Add,
        );

        let stencil = wgpu::StencilState {
            front: wgpu::StencilFaceState::default(),
            back: wgpu::StencilFaceState::default(),
            read_mask: 0xFF,
            write_mask: 0xFF,
        };
        let depth = DepthStencilState::new(
            wgpu::TextureFormat::Depth32Float,
            true,
            wgpu::CompareFunction::Less,
            stencil,
        );

        let pipeline = create_render_pipeline(
            &device,
            None,
            vs_src,
            fs_src,
            &[],
            Some(blend),
            Some(depth),
            wgpu::TextureFormat::Rgba8Unorm,
        );

        assert!(pipeline.is_ok(), "pipeline with depth+blend should compile");
    }
}
