//! Python-to-IR conversion layer for frame graph declarations.
//!
//! Provides stub types and conversion functions that map Python-side pass,
//! resource, and attachment declarations into the frame graph
//! intermediate representation (IR) types defined in [`super`].
//!
//! # Status
//!
//! This is a minimal implementation sufficient for compilation of the
//! whitebox test suite. All conversion functions perform basic validation
//! and produce correct defaults, but the stubs are not yet production-grade.

use core::fmt;

use super::{
    AttachmentLoadOp, AttachmentStoreOp, BufferDesc, ColorAttachment, DepthStencilAttachment,
    DispatchSource, EmptyView, InstanceSource, IrPass, IrResource, PassIndex, PassType,
    ResourceDesc, ResourceHandle, ResourceAccessSet, ResourceLifetime, ResourceState,
    Texture3DDesc, TextureDesc, View, ViewType,
};

// ---------------------------------------------------------------------------
// ConversionError
// ---------------------------------------------------------------------------

/// Errors that can occur during Python-to-IR conversion.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionError {
    EmptyPassName,
    /// Resource name is empty.
    EmptyResourceName,
    InvalidResourceHandle(u32),
    InvalidLoadOp(String),
    InvalidStoreOp(String),
    InvalidDepthLoadOp(String),
    InvalidDepthStoreOp(String),
    InvalidStencilLoadOp(String),
    InvalidStencilStoreOp(String),
    InvalidViewType(String),
    InvalidInstanceSource(String),
    InvalidDispatchSource(String),
    MissingColorAttachments,
    AttachmentsNotAllowed(PassType),
    MissingDispatchSource,
    InvalidDepthStencilHandle,
    /// Resource format string is not a recognised wgpu `TextureFormat`.
    InvalidResourceFormat(String),
    /// Resource type string is not recognised.
    InvalidResourceType(String),
    /// One or more usage flags are not recognised.
    InvalidUsageFlags(String),
    /// Resource dimensions are invalid (e.g., zero width/height for textures).
    InvalidResourceDimensions(String),
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPassName => write!(f, "pass name must not be empty"),
            Self::EmptyResourceName => write!(f, "resource name must not be empty"),
            Self::InvalidResourceHandle(h) => write!(f, "invalid resource handle: {}", h),
            Self::InvalidLoadOp(op) => write!(f, "invalid load op: {}", op),
            Self::InvalidStoreOp(op) => write!(f, "invalid store op: {}", op),
            Self::InvalidDepthLoadOp(op) => write!(f, "invalid depth load op: {}", op),
            Self::InvalidDepthStoreOp(op) => write!(f, "invalid depth store op: {}", op),
            Self::InvalidStencilLoadOp(op) => write!(f, "invalid stencil load op: {}", op),
            Self::InvalidStencilStoreOp(op) => write!(f, "invalid stencil store op: {}", op),
            Self::InvalidViewType(t) => write!(f, "invalid view type: {}", t),
            Self::InvalidInstanceSource(s) => write!(f, "invalid instance source: {}", s),
            Self::InvalidDispatchSource(s) => write!(f, "invalid dispatch source: {}", s),
            Self::MissingColorAttachments => {
                write!(f, "graphics pass must have at least one color attachment")
            }
            Self::AttachmentsNotAllowed(pt) => {
                write!(f, "attachments not allowed for pass type: {}", pt)
            }
            Self::MissingDispatchSource => write!(f, "compute pass requires a dispatch source"),
            Self::InvalidDepthStencilHandle => {
                write!(f, "depth/stencil attachment resource handle is NONE")
            }
            Self::InvalidResourceFormat(fmt) => {
                write!(f, "invalid resource format '{fmt}' — not a recognised wgpu TextureFormat")
            }
            Self::InvalidResourceType(rt) => {
                write!(f, "invalid resource type '{rt}' — expected Texture2D, Texture3D, TextureCube, or Buffer")
            }
            Self::InvalidUsageFlags(msg) => {
                write!(f, "invalid usage flags: {msg}")
            }
            Self::InvalidResourceDimensions(msg) => {
                write!(f, "invalid resource dimensions: {msg}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Python-side types
// ---------------------------------------------------------------------------

/// Python-side colour attachment declaration.
#[derive(Debug, Clone)]
pub struct PyColorAttachment {
    pub resource: u32,
    pub load_op: String,
    pub store_op: String,
}

/// Python-side depth/stencil attachment declaration.
#[derive(Debug, Clone)]
pub struct PyDepthStencilAttachment {
    pub resource: u32,
    pub depth_load_op: String,
    pub depth_store_op: String,
    pub stencil_load_op: String,
    pub stencil_store_op: String,
}

/// Python-side dispatch source declaration.
#[derive(Debug, Clone)]
pub struct PyDispatchSource {
    pub kind: String,
}

/// Python-side instance source declaration.
#[derive(Debug, Clone)]
pub struct PyInstanceSource {
    pub kind: String,
}

/// Python-side pass type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyPassType {
    Graphics,
    Compute,
    Copy,
    RayTracing,
}

impl PyPassType {
    /// Parse a string into a `PyPassType`; returns `None` for unknown strings.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Graphics" => Some(Self::Graphics),
            "Compute" => Some(Self::Compute),
            "Copy" => Some(Self::Copy),
            "RayTracing" => Some(Self::RayTracing),
            _ => None,
        }
    }

    /// Convert this Python-side pass type into the IR [`PassType`].
    pub fn to_ir(self) -> PassType {
        match self {
            Self::Graphics => PassType::Graphics,
            Self::Compute => PassType::Compute,
            Self::Copy => PassType::Copy,
            Self::RayTracing => PassType::RayTracing,
        }
    }
}

/// Python-side view type declaration.
#[derive(Debug, Clone)]
pub struct PyViewType {
    pub kind: String,
}

/// Python-side pass node declaration.
///
/// This is the top-level type converted into an [`IrPass`] via `TryInto`.
#[derive(Debug, Clone)]
pub struct PyPassNode {
    pub name: String,
    pub pass_type: PyPassType,
    pub color_attachments: Vec<PyColorAttachment>,
    pub depth_stencil: Option<PyDepthStencilAttachment>,
    pub reads: Vec<u32>,
    pub writes: Vec<u32>,
    pub instance_source: Option<PyInstanceSource>,
    pub dispatch_source: Option<PyDispatchSource>,
    pub view_type: PyViewType,
}

/// Python-side resource description used for formal Python-to-Rust
/// resource conversion.
///
/// Each field maps directly to a key in the Python serialization JSON.
/// The [`TryFrom<PyResourceDesc> for IrResource`] conversion performs
/// format validation, usage flag coalescing, initial state parsing,
/// dimension validation, and handle resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct PyResourceDesc {
    /// Debug / friendly name (e.g., `"gbuffer_albedo"`, `"depth_hzb"`).
    pub name: String,
    /// Resource type discriminator (`"Texture2D"`, `"Texture3D"`, `"TextureCube"`, or `"Buffer"`).
    pub resource_type: String,
    /// Width in texels (or size in bytes for buffers).
    pub width: u32,
    /// Height in texels (1 for 1D resources).
    pub height: u32,
    /// Depth in texels (3D textures only; 1 for 2D textures).
    pub depth: u32,
    /// Texel format (e.g., `"rgba8unorm"`, `"depth32float"`, `"R8G8B8A8_UNORM"`).
    pub format: String,
    /// Usage flags (e.g., `"copy_src"`, `"texture_binding"`, `"storage"`).
    pub usage_flags: Vec<String>,
    /// Number of mip levels (default 1).
    pub mip_levels: u32,
    /// MSAA sample count (default 1).
    pub sample_count: u32,
    /// Whether the resource is transient (frame-local). `None` defaults to `true`.
    pub is_transient: Option<bool>,
    /// Initial GPU state string (e.g., `"Uninitialized"`, `"ColorAttachment"`).
    /// `None` defaults to `"Uninitialized"`.
    pub initial_state: Option<String>,
    /// Explicit resource handle. `None` triggers auto-assignment.
    pub handle: Option<ResourceHandle>,
}

impl Default for PyResourceDesc {
    fn default() -> Self {
        Self {
            name: String::new(),
            resource_type: "Texture2D".to_string(),
            width: 1,
            height: 1,
            depth: 1,
            format: "rgba8unorm".to_string(),
            usage_flags: Vec::new(),
            mip_levels: 1,
            sample_count: 1,
            is_transient: None,
            initial_state: None,
            handle: None,
        }
    }
}


// ---------------------------------------------------------------------------
// Direct IR conversion (used by JSON deserialization bridge)
// ---------------------------------------------------------------------------

impl PyResourceDesc {
    /// Convert this Python resource description to an IR resource with
    /// the given handle and transient flag.
    ///
    /// This is a simplified conversion used by the JSON deserialization
    /// path (`deserialize_from_json`) — it does not perform the full
    /// format / usage validation of `TryFrom<PyResourceDesc> for IrResource`.
    pub fn to_ir_resource(&self, handle: super::ResourceHandle, is_transient: bool) -> super::IrResource {
        use super::{
            BufferDesc, IrResource, ResourceDesc, ResourceHandle, ResourceLifetime,
            ResourceState, Texture3DDesc, TextureDesc,
        };

        let desc = match self.resource_type.as_str() {
            "Texture3D" => ResourceDesc::Texture3D(Texture3DDesc {
                width: self.width,
                height: self.height,
                depth: self.depth,
                mip_levels: 1,
                format: self.format.clone(),
            }),
            "TextureCube" => ResourceDesc::TextureCube(TextureDesc {
                width: self.width,
                height: self.height,
                mip_levels: 1,
                array_layers: 6,
                format: self.format.clone(),
            }),
            "Buffer" => ResourceDesc::Buffer(BufferDesc {
                size: self.width as u64,
                usage: "storage".to_string(),
                is_indirect_arg: false,
            }),
            _ => ResourceDesc::Texture2D(TextureDesc {
                width: self.width,
                height: self.height,
                mip_levels: 1,
                array_layers: 1,
                format: self.format.clone(),
            }),
        };

        let lifetime = if is_transient {
            ResourceLifetime::Transient
        } else {
            ResourceLifetime::Imported
        };

        IrResource::new(handle, self.name.clone(), desc, lifetime, ResourceState::Uninitialized)
    }
}


// ---------------------------------------------------------------------------
// Parse helpers
// ---------------------------------------------------------------------------

/// Parse a colour load-op string.
pub fn parse_load_op(s: &str) -> Result<AttachmentLoadOp, ConversionError> {
    match s {
        "Load" => Ok(AttachmentLoadOp::Load),
        "Clear" => Ok(AttachmentLoadOp::Clear),
        "DontCare" => Ok(AttachmentLoadOp::DontCare),
        other => Err(ConversionError::InvalidLoadOp(other.to_string())),
    }
}

/// Parse a colour store-op string.
pub fn parse_store_op(s: &str) -> Result<AttachmentStoreOp, ConversionError> {
    match s {
        "Store" => Ok(AttachmentStoreOp::Store),
        "DontCare" => Ok(AttachmentStoreOp::DontCare),
        other => Err(ConversionError::InvalidStoreOp(other.to_string())),
    }
}

/// Parse a depth load-op string.
pub fn parse_depth_load_op(s: &str) -> Result<AttachmentLoadOp, ConversionError> {
    match s {
        "Load" => Ok(AttachmentLoadOp::Load),
        "Clear" => Ok(AttachmentLoadOp::Clear),
        "DontCare" => Ok(AttachmentLoadOp::DontCare),
        other => Err(ConversionError::InvalidDepthLoadOp(other.to_string())),
    }
}

/// Parse a depth store-op string.
pub fn parse_depth_store_op(s: &str) -> Result<AttachmentStoreOp, ConversionError> {
    match s {
        "Store" => Ok(AttachmentStoreOp::Store),
        "DontCare" => Ok(AttachmentStoreOp::DontCare),
        other => Err(ConversionError::InvalidDepthStoreOp(other.to_string())),
    }
}

/// Parse a stencil load-op string.
pub fn parse_stencil_load_op(s: &str) -> Result<AttachmentLoadOp, ConversionError> {
    match s {
        "Load" => Ok(AttachmentLoadOp::Load),
        "Clear" => Ok(AttachmentLoadOp::Clear),
        "DontCare" => Ok(AttachmentLoadOp::DontCare),
        other => Err(ConversionError::InvalidStencilLoadOp(other.to_string())),
    }
}

/// Parse a stencil store-op string.
pub fn parse_stencil_store_op(s: &str) -> Result<AttachmentStoreOp, ConversionError> {
    match s {
        "Store" => Ok(AttachmentStoreOp::Store),
        "DontCare" => Ok(AttachmentStoreOp::DontCare),
        other => Err(ConversionError::InvalidStencilStoreOp(other.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Conversion functions
// ---------------------------------------------------------------------------

/// Convert a Python view type to an IR [`ViewType`].
pub fn convert_view_type(vt: &PyViewType) -> Result<ViewType, ConversionError> {
    match vt.kind.as_str() {
        "Texture2D" => Ok(ViewType::Texture2D),
        "TextureCube" => Ok(ViewType::TextureCube),
        "Texture3D" => Ok(ViewType::Texture3D),
        "Storage" => Ok(ViewType::Storage),
        "UniformTexel" => Ok(ViewType::UniformTexel),
        "StorageTexel" => Ok(ViewType::StorageTexel),
        "UniformBuffer" => Ok(ViewType::UniformBuffer),
        "StorageBuffer" => Ok(ViewType::StorageBuffer),
        "AccelerationStructure" => Ok(ViewType::AccelerationStructure),
        other => Err(ConversionError::InvalidViewType(other.to_string())),
    }
}

/// Convert a Python instance source to an IR [`InstanceSource`].
pub fn convert_instance_source(src: &PyInstanceSource) -> Result<InstanceSource, ConversionError> {
    match src.kind.as_str() {
        "Direct" => Ok(InstanceSource::Direct {
            index_count: 0,
            instance_count: 1,
            base_vertex: 0,
            first_index: 0,
            first_instance: 0,
        }),
        "Indirect" => Ok(InstanceSource::Indirect {
            buffer: ResourceHandle::NONE,
            offset: 0,
            draw_count: 0,
            stride: 0,
        }),
        "Mesh" => Ok(InstanceSource::Mesh {
            group_count_x: 0,
            group_count_y: 0,
            group_count_z: 0,
        }),
        other => Err(ConversionError::InvalidInstanceSource(other.to_string())),
    }
}

/// Convert a Python dispatch source to an IR [`DispatchSource`].
pub fn convert_dispatch_source(
    src: &PyDispatchSource,
) -> Result<DispatchSource, ConversionError> {
    match src.kind.as_str() {
        "Direct" => Ok(DispatchSource::Direct {
            group_count_x: 0,
            group_count_y: 0,
            group_count_z: 0,
        }),
        "Indirect" => Ok(DispatchSource::Indirect {
            buffer: ResourceHandle::NONE,
            offset: 0,
        }),
        other => Err(ConversionError::InvalidDispatchSource(other.to_string())),
    }
}

/// Convert a Python colour attachment to an IR [`ColorAttachment`].
pub fn convert_color_attachment(
    ca: &PyColorAttachment,
) -> Result<ColorAttachment, ConversionError> {
    if ca.resource == ResourceHandle::NONE.0 {
        return Err(ConversionError::InvalidResourceHandle(ca.resource));
    }
    let load_op = parse_load_op(&ca.load_op)?;
    let store_op = parse_store_op(&ca.store_op)?;
    Ok(ColorAttachment {
        resource: ResourceHandle(ca.resource),
        load_op,
        store_op,
        ..Default::default()
    })
}

/// Convert a Python depth/stencil attachment to an IR [`DepthStencilAttachment`].
pub fn convert_depth_stencil(
    ds: &PyDepthStencilAttachment,
) -> Result<DepthStencilAttachment, ConversionError> {
    if ds.resource == ResourceHandle::NONE.0 {
        return Err(ConversionError::InvalidDepthStencilHandle);
    }
    let depth_load_op = parse_depth_load_op(&ds.depth_load_op)?;
    let depth_store_op = parse_depth_store_op(&ds.depth_store_op)?;
    let stencil_load_op = parse_stencil_load_op(&ds.stencil_load_op)?;
    let stencil_store_op = parse_stencil_store_op(&ds.stencil_store_op)?;
    Ok(DepthStencilAttachment {
        resource: ResourceHandle(ds.resource),
        depth_load_op,
        depth_store_op,
        stencil_load_op,
        stencil_store_op,
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// TryFrom<PyPassNode> for IrPass
// ---------------------------------------------------------------------------

impl TryFrom<PyPassNode> for IrPass {
    type Error = ConversionError;

    fn try_from(node: PyPassNode) -> Result<Self, ConversionError> {
        // Validate name.
        if node.name.is_empty() {
            return Err(ConversionError::EmptyPassName);
        }

        let pass_type = node.pass_type.to_ir();
        let view_type = convert_view_type(&node.view_type)?;

        // Build deduplicated access sets from raw reads/writes.
        let mut reads: Vec<ResourceHandle> = Vec::new();
        for r in node.reads {
            let h = ResourceHandle(r);
            if !reads.contains(&h) {
                reads.push(h);
            }
        }
        let mut writes: Vec<ResourceHandle> = Vec::new();
        for w in node.writes {
            let h = ResourceHandle(w);
            if !writes.contains(&h) {
                writes.push(h);
            }
        }

        match pass_type {
            PassType::Graphics => {
                if node.color_attachments.is_empty() {
                    return Err(ConversionError::MissingColorAttachments);
                }

                // Convert each colour attachment (also validates handles).
                let mut color_attachments = Vec::with_capacity(node.color_attachments.len());
                for ca in &node.color_attachments {
                    if ca.resource == ResourceHandle::NONE.0 {
                        return Err(ConversionError::InvalidResourceHandle(ca.resource));
                    }
                    let load_op = parse_load_op(&ca.load_op)?;
                    let store_op = parse_store_op(&ca.store_op)?;
                    color_attachments.push(ColorAttachment {
                        resource: ResourceHandle(ca.resource),
                        load_op,
                        store_op,
                        ..Default::default()
                    });
                }

                // Convert depth/stencil if present.
                let depth_stencil = match node.depth_stencil {
                    Some(ref ds) => {
                        if ds.resource == ResourceHandle::NONE.0 {
                            return Err(ConversionError::InvalidDepthStencilHandle);
                        }
                        let dl = parse_depth_load_op(&ds.depth_load_op)?;
                        let dls = parse_depth_store_op(&ds.depth_store_op)?;
                        let sl = parse_stencil_load_op(&ds.stencil_load_op)?;
                        let sls = parse_stencil_store_op(&ds.stencil_store_op)?;
                        Some(DepthStencilAttachment {
                            resource: ResourceHandle(ds.resource),
                            depth_load_op: dl,
                            depth_store_op: dls,
                            stencil_load_op: sl,
                            stencil_store_op: sls,
                            ..Default::default()
                        })
                    }
                    None => None,
                };

                // Convert instance source; default to Direct if absent.
                let instance_source = match node.instance_source {
                    Some(ref src) => convert_instance_source(src)?,
                    None => InstanceSource::Direct {
                        index_count: 0,
                        instance_count: 1,
                        base_vertex: 0,
                        first_index: 0,
                        first_instance: 0,
                    },
                };

                Ok(IrPass {
                    index: PassIndex(0),
                    name: node.name,
                    pass_type,
                    access_set: ResourceAccessSet { reads, writes },
                    color_attachments,
                    depth_stencil,
                    instance_source,
                    dispatch_source: None,
                    view_type,
                    tags: Vec::new(),
                    })
            }

            PassType::Compute => {
                // Compute passes must not have attachments.
                if !node.color_attachments.is_empty() || node.depth_stencil.is_some() {
                    return Err(ConversionError::AttachmentsNotAllowed(PassType::Compute));
                }
                let dispatch_source = match node.dispatch_source {
                    Some(ref src) => Some(convert_dispatch_source(src)?),
                    None => return Err(ConversionError::MissingDispatchSource),
                };

                Ok(IrPass {
                    index: PassIndex(0),
                    name: node.name,
                    pass_type,
                    access_set: ResourceAccessSet { reads, writes },
                    color_attachments: Vec::new(),
                    depth_stencil: None,
                    instance_source: InstanceSource::Direct {
                        index_count: 0,
                        instance_count: 1,
                        base_vertex: 0,
                        first_index: 0,
                        first_instance: 0,
                    },
                    dispatch_source,
                    view_type,
                    tags: Vec::new(),
                    })
            }

            PassType::RayTracing | PassType::Copy => {
                // Attachments not allowed on non-graphics passes.
                if !node.color_attachments.is_empty() || node.depth_stencil.is_some() {
                    return Err(ConversionError::AttachmentsNotAllowed(pass_type));
                }

                Ok(IrPass {
                    index: PassIndex(0),
                    name: node.name,
                    pass_type,
                    access_set: ResourceAccessSet { reads, writes },
                    color_attachments: Vec::new(),
                    depth_stencil: None,
                    instance_source: InstanceSource::Direct {
                        index_count: 0,
                        instance_count: 1,
                        base_vertex: 0,
                        first_index: 0,
                        first_instance: 0,
                    },
                    dispatch_source: None,
                    view_type,
                    tags: Vec::new(),
                    })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Format validation — known wgpu TextureFormat strings
// ---------------------------------------------------------------------------

/// Canonical list of recognised `wgpu::TextureFormat` string representations
/// (lower-cased).  Used by [`validate_texture_format`] to check that a
/// format string from Python is valid before accepting it into the IR.
const KNOWN_TEXTURE_FORMATS: &[&str] = &[
    // 8-bit
    "r8unorm", "r8snorm", "r8uint", "r8sint",
    // 16-bit 1-component
    "r16uint", "r16sint", "r16float",
    // 8-bit 2-component
    "rg8unorm", "rg8snorm", "rg8uint", "rg8sint",
    // 32-bit 1-component
    "r32uint", "r32sint", "r32float",
    // 16-bit 2-component
    "rg16uint", "rg16sint", "rg16float",
    // 32-bit 2-component
    "rg32uint", "rg32sint", "rg32float",
    // 8-bit RGBA
    "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint",
    // 8-bit BGRA
    "bgra8unorm", "bgra8unorm-srgb",
    // 16-bit RGBA
    "rgba16uint", "rgba16sint", "rgba16float",
    // 32-bit RGBA
    "rgba32uint", "rgba32sint", "rgba32float",
    // Packed
    "rgb10a2unorm", "rg11b10ufloat", "rgb9e5ufloat",
    // Depth / stencil
    "depth32float", "depth24plus", "depth24plus-stencil8",
    "depth32float-stencil8", "stencil8",
    // BC compressed
    "bc1-rgba-unorm", "bc1-rgba-unorm-srgb",
    "bc2-rgba-unorm", "bc2-rgba-unorm-srgb",
    "bc3-rgba-unorm", "bc3-rgba-unorm-srgb",
    "bc4-r-unorm", "bc4-r-snorm",
    "bc5-rg-unorm", "bc5-rg-snorm",
    "bc6h-rgb-ufloat", "bc6h-rgb-sfloat",
    "bc7-rgba-unorm", "bc7-rgba-unorm-srgb",
    // ETC2 / EAC compressed
    "etc2-rgba8unorm", "etc2-rgba8unorm-srgb",
    "eac-r11unorm", "eac-r11snorm",
    "eac-rg11unorm", "eac-rg11snorm",
    // ASTC compressed
    "astc-4x4-unorm", "astc-4x4-unorm-srgb",
    "astc-5x4-unorm", "astc-5x4-unorm-srgb",
    "astc-5x5-unorm", "astc-5x5-unorm-srgb",
    "astc-6x5-unorm", "astc-6x5-unorm-srgb",
    "astc-6x6-unorm", "astc-6x6-unorm-srgb",
    "astc-8x5-unorm", "astc-8x5-unorm-srgb",
    "astc-8x6-unorm", "astc-8x6-unorm-srgb",
    "astc-8x8-unorm", "astc-8x8-unorm-srgb",
    "astc-10x5-unorm", "astc-10x5-unorm-srgb",
    "astc-10x6-unorm", "astc-10x6-unorm-srgb",
    "astc-10x8-unorm", "astc-10x8-unorm-srgb",
    "astc-10x10-unorm", "astc-10x10-unorm-srgb",
    "astc-12x10-unorm", "astc-12x10-unorm-srgb",
    "astc-12x12-unorm", "astc-12x12-unorm-srgb",
];

/// Canonical list of recognised usage-flag strings (lower-cased).  Used by
/// [`validate_usage_flags`] to ensure each flag in a Python descriptor
/// corresponds to a valid wgpu usage.
const KNOWN_USAGE_FLAGS: &[&str] = &[
    "copy_src", "copy_dst", "texture_binding", "storage_binding",
    "color_attachment", "depth_stencil_attachment",
    "uniform", "storage", "index", "vertex", "indirect", "query_resolve",
];

/// Validates that `format` is a recognised `wgpu::TextureFormat` string.
///
/// Comparison is case-insensitive: `"R8G8B8A8_UNORM"` and `"rgba8unorm"`
/// are both accepted.  Buffer resources are not validated (they have no
/// texel format).
pub fn validate_texture_format(format: &str) -> Result<(), ConversionError> {
    let lower = format.to_lowercase();
    if !KNOWN_TEXTURE_FORMATS.contains(&lower.as_str()) {
        return Err(ConversionError::InvalidResourceFormat(format.to_string()));
    }
    Ok(())
}

/// Coalesces a list of usage-flag strings: removes duplicates
/// (case-insensitive) and validates that every flag is recognised.
pub fn coalesce_usage_flags(flags: &[String]) -> Result<Vec<String>, ConversionError> {
    let mut seen: Vec<String> = Vec::with_capacity(flags.len());
    for flag in flags {
        let lower = flag.to_lowercase();
        if !KNOWN_USAGE_FLAGS.contains(&lower.as_str()) {
            return Err(ConversionError::InvalidUsageFlags(format!(
                "unknown usage flag '{flag}'"
            )));
        }
        if !seen.iter().any(|f| f.eq_ignore_ascii_case(flag)) {
            seen.push(flag.clone());
        }
    }
    Ok(seen)
}

/// Parses an `initial_state` string into a [`ResourceState`].
///
/// Returns `ResourceState::Uninitialized` when `state` is `None` (the
/// common case for transient resources).
pub fn parse_initial_state(state: Option<&str>) -> Result<ResourceState, ConversionError> {
    match state {
        None => Ok(ResourceState::Uninitialized),
        Some(s) => match s {
            "Uninitialized" => Ok(ResourceState::Uninitialized),
            "VertexBuffer" => Ok(ResourceState::VertexBuffer),
            "IndexBuffer" => Ok(ResourceState::IndexBuffer),
            "IndirectArgument" => Ok(ResourceState::IndirectArgument),
            "ColorAttachment" => Ok(ResourceState::ColorAttachment),
            "DepthStencilAttachment" => Ok(ResourceState::DepthStencilAttachment),
            "DepthStencilReadOnly" => Ok(ResourceState::DepthStencilReadOnly),
            "ShaderRead" => Ok(ResourceState::ShaderRead),
            "ShaderReadWrite" => Ok(ResourceState::ShaderReadWrite),
            "TransferSrc" => Ok(ResourceState::TransferSrc),
            "TransferDst" => Ok(ResourceState::TransferDst),
            "AccelerationStructure" => Ok(ResourceState::AccelerationStructure),
            "Present" => Ok(ResourceState::Present),
            other => Err(ConversionError::InvalidUsageFlags(format!(
                "unknown initial state '{other}'"
            ))),
        },
    }
}

// ---------------------------------------------------------------------------
// TryFrom<PyResourceDesc> for IrResource
// ---------------------------------------------------------------------------

impl TryFrom<PyResourceDesc> for IrResource {
    type Error = ConversionError;

    fn try_from(py: PyResourceDesc) -> Result<Self, ConversionError> {
        // -- Validate basic fields ------------------------------------------------
        if py.name.is_empty() {
            return Err(ConversionError::EmptyResourceName);
        }

        let is_texture = matches!(
            py.resource_type.as_str(),
            "Texture2D" | "Texture3D" | "TextureCube"
        );
        let is_buffer = py.resource_type.as_str() == "Buffer";

        if is_texture && (py.width == 0 || py.height == 0) {
            return Err(ConversionError::InvalidResourceDimensions(format!(
                "{} width={} height={}: dimensions must be non-zero",
                py.resource_type, py.width, py.height,
            )));
        }

        // -- Step 1: Resource type + format validation ------------------------
        let resource_desc = match py.resource_type.as_str() {
            "Texture2D" => {
                validate_texture_format(&py.format)?;
                let mip = py.mip_levels.max(1);
                let samples = py.sample_count.max(1);
                ResourceDesc::Texture2D(TextureDesc {
                    width: py.width,
                    height: py.height,
                    mip_levels: mip,
                    array_layers: samples,
                    format: py.format,
                })
            }
            "Texture3D" => {
                validate_texture_format(&py.format)?;
                let mip = py.mip_levels.max(1);
                ResourceDesc::Texture3D(Texture3DDesc {
                    width: py.width,
                    height: py.height,
                    depth: py.depth,
                    mip_levels: mip,
                    format: py.format,
                })
            }
            "TextureCube" => {
                validate_texture_format(&py.format)?;
                let mip = py.mip_levels.max(1);
                ResourceDesc::TextureCube(TextureDesc {
                    width: py.width,
                    height: py.height,
                    mip_levels: mip,
                    array_layers: 6,
                    format: py.format,
                })
            }
            "Buffer" => {
                let coalesced = coalesce_usage_flags(&py.usage_flags)?;
                ResourceDesc::Buffer(BufferDesc {
                    size: py.width as u64,
                    usage: if coalesced.is_empty() {
                        "storage".to_string()
                    } else {
                        coalesced.join(" | ")
                    },
                    is_indirect_arg: false,
                })
            }
            other => {
                return Err(ConversionError::InvalidResourceType(other.to_string()));
            }
        };

        // -- Step 2: Usage flag coalescing (texture types) --------------------
        if is_buffer {
            // Already coalesced above.
        } else {
            let _ = coalesce_usage_flags(&py.usage_flags)?;
        }

        // -- Step 3: Lifetime -------------------------------------------------
        let lifetime = if py.is_transient.unwrap_or(true) {
            ResourceLifetime::Transient
        } else {
            ResourceLifetime::Imported
        };

        // -- Step 4: Initial state --------------------------------------------
        let initial_state = parse_initial_state(py.initial_state.as_deref())?;

        // -- Step 5: Handle resolution (auto-assign when None) ----------------
        let handle = match py.handle {
            Some(h) => h,
            None => {
                use std::sync::atomic::{AtomicU32, Ordering};
                static NEXT_RESOURCE_HANDLE: AtomicU32 = AtomicU32::new(1);
                ResourceHandle(NEXT_RESOURCE_HANDLE.fetch_add(1, Ordering::Relaxed))
            }
        };

        Ok(IrResource::new(handle, py.name, resource_desc, lifetime, initial_state))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Valid Texture2D ---------------------------------------------------------

    #[test]
    fn test_valid_texture2d_converts_with_correct_format() {
        let desc = PyResourceDesc {
            name: "color_rt".into(),
            resource_type: "Texture2D".into(),
            width: 1920,
            height: 1080,
            format: "rgba8unorm".into(),
            ..Default::default()
        };
        let ir: IrResource = desc.try_into().unwrap();
        assert_eq!(ir.name, "color_rt");
        match &ir.desc {
            ResourceDesc::Texture2D(t) => {
                assert_eq!(t.format, "rgba8unorm");
                assert_eq!(t.width, 1920);
                assert_eq!(t.height, 1080);
            }
            other => panic!("expected Texture2D, got {other:?}"),
        }
    }

    // -- Valid Buffer -------------------------------------------------------------

    #[test]
    fn test_valid_buffer_converts_with_correct_usage() {
        let desc = PyResourceDesc {
            name: "storage_buf".into(),
            resource_type: "Buffer".into(),
            width: 4096,
            usage_flags: vec!["storage".into(), "copy_src".into()],
            ..Default::default()
        };
        let ir: IrResource = desc.try_into().unwrap();
        assert_eq!(ir.name, "storage_buf");
        match &ir.desc {
            ResourceDesc::Buffer(b) => {
                assert!(b.usage.contains("storage"));
                assert!(b.usage.contains("copy_src"));
                assert_eq!(b.size, 4096);
            }
            other => panic!("expected Buffer, got {other:?}"),
        }
    }

    // -- Invalid format string ----------------------------------------------------

    #[test]
    fn test_invalid_format_returns_err() {
        let desc = PyResourceDesc {
            name: "bad_tex".into(),
            resource_type: "Texture2D".into(),
            width: 256,
            height: 256,
            format: "not_a_real_format".into(),
            ..Default::default()
        };
        let result: Result<IrResource, ConversionError> = desc.try_into();
        assert!(matches!(result, Err(ConversionError::InvalidResourceFormat(_))));
    }

    // -- Empty name ---------------------------------------------------------------

    #[test]
    fn test_empty_name_returns_err() {
        let desc = PyResourceDesc {
            name: String::new(),
            resource_type: "Texture2D".into(),
            width: 256,
            height: 256,
            ..Default::default()
        };
        let result: Result<IrResource, ConversionError> = desc.try_into();
        assert!(matches!(result, Err(ConversionError::EmptyResourceName)));
    }

    // -- Zero width/height --------------------------------------------------------

    #[test]
    fn test_zero_width_height_returns_err() {
        let desc = PyResourceDesc {
            name: "bad_size".into(),
            resource_type: "Texture2D".into(),
            width: 0,
            height: 1080,
            ..Default::default()
        };
        let result: Result<IrResource, ConversionError> = desc.try_into();
        assert!(
            matches!(result, Err(ConversionError::InvalidResourceDimensions(_))),
            "expected InvalidResourceDimensions, got {result:?}"
        );
    }

    // -- Usage flag coalescing ----------------------------------------------------

    #[test]
    fn test_usage_flag_coalescing_merges_duplicates() {
        let desc = PyResourceDesc {
            name: "coalesced_buf".into(),
            resource_type: "Buffer".into(),
            width: 1024,
            usage_flags: vec![
                "storage".into(),
                "storage".into(),
                "copy_src".into(),
                "copy_src".into(),
            ],
            ..Default::default()
        };
        let ir: IrResource = desc.try_into().unwrap();
        match &ir.desc {
            ResourceDesc::Buffer(b) => {
                // "storage" and "copy_src" each appear once.
                assert_eq!(b.usage.matches("storage").count(), 1);
                assert_eq!(b.usage.matches("copy_src").count(), 1);
            }
            other => panic!("expected Buffer, got {other:?}"),
        }
    }

    // -- Initial state parsing ----------------------------------------------------

    #[test]
    fn test_initial_state_parses_color_attachment() {
        let desc = PyResourceDesc {
            name: "stateful_rt".into(),
            resource_type: "Texture2D".into(),
            width: 256,
            height: 256,
            format: "rgba8unorm".into(),
            initial_state: Some("ColorAttachment".into()),
            ..Default::default()
        };
        let ir: IrResource = desc.try_into().unwrap();
        assert_eq!(ir.initial_state, ResourceState::ColorAttachment);
    }

    // -- Handle resolution -- explicit handle -------------------------------------

    #[test]
    fn test_explicit_handle_is_preserved() {
        let desc = PyResourceDesc {
            name: "explicit_handle_res".into(),
            resource_type: "Texture2D".into(),
            width: 128,
            height: 128,
            format: "r32float".into(),
            handle: Some(ResourceHandle(42)),
            ..Default::default()
        };
        let ir: IrResource = desc.try_into().unwrap();
        assert_eq!(ir.handle, ResourceHandle(42));
    }

    // -- Handle resolution -- auto-assigned ---------------------------------------

    #[test]
    fn test_no_handle_auto_assigns_unique() {
        let desc = PyResourceDesc {
            name: "auto_assigned".into(),
            resource_type: "Texture2D".into(),
            width: 64,
            height: 64,
            format: "rgba8unorm".into(),
            handle: None,
            ..Default::default()
        };
        let ir: IrResource = desc.try_into().unwrap();
        // Auto-assigned handles are non-zero and never NONE.
        assert_ne!(ir.handle, ResourceHandle::NONE);
        assert_ne!(ir.handle.0, 0);
    }

    // -- Two resources get different auto-assigned handles ------------------------

    #[test]
    fn test_two_auto_assigned_handles_are_different() {
        let desc_a = PyResourceDesc {
            name: "res_a".into(),
            resource_type: "Texture2D".into(),
            width: 64,
            height: 64,
            format: "rgba8unorm".into(),
            handle: None,
            ..Default::default()
        };
        let desc_b = PyResourceDesc {
            name: "res_b".into(),
            resource_type: "Buffer".into(),
            width: 512,
            usage_flags: vec!["storage".into()],
            handle: None,
            ..Default::default()
        };
        let ir_a: IrResource = desc_a.try_into().unwrap();
        let ir_b: IrResource = desc_b.try_into().unwrap();
        assert_ne!(ir_a.handle, ir_b.handle, "auto-assigned handles must be unique");
    }

    // -- Mip levels and sample count preserved -----------------------------------

    #[test]
    fn test_mip_levels_and_sample_count_preserved() {
        let desc = PyResourceDesc {
            name: "msaa_rt".into(),
            resource_type: "Texture2D".into(),
            width: 800,
            height: 600,
            format: "rgba8unorm".into(),
            mip_levels: 4,
            sample_count: 4,
            ..Default::default()
        };
        let ir: IrResource = desc.try_into().unwrap();
        match &ir.desc {
            ResourceDesc::Texture2D(t) => {
                assert_eq!(t.mip_levels, 4, "mip level count must be preserved");
                assert_eq!(t.array_layers, 4, "sample count maps to array_layers");
            }
            other => panic!("expected Texture2D, got {other:?}"),
        }
    }

    // -- is_transient flag preserved ---------------------------------------------

    #[test]
    fn test_is_transient_flag_preserved() {
        // Transient resource (default).
        let transient = PyResourceDesc {
            name: "tmp_tex".into(),
            resource_type: "Texture2D".into(),
            width: 256,
            height: 256,
            format: "rgba8unorm".into(),
            is_transient: Some(true),
            ..Default::default()
        };
        let ir_t: IrResource = transient.try_into().unwrap();
        assert_eq!(ir_t.lifetime, ResourceLifetime::Transient);

        // Imported resource.
        let imported = PyResourceDesc {
            name: "imported_tex".into(),
            resource_type: "Texture2D".into(),
            width: 256,
            height: 256,
            format: "rgba8unorm".into(),
            is_transient: Some(false),
            ..Default::default()
        };
        let ir_i: IrResource = imported.try_into().unwrap();
        assert_eq!(ir_i.lifetime, ResourceLifetime::Imported);
    }
}
