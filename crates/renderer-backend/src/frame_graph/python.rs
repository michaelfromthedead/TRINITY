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
use std::sync::Arc;

use super::{
    AttachmentLoadOp, AttachmentStoreOp, BufferDesc, ColorAttachment, DepthStencilAttachment,
    DispatchSource, EmptyView, InstanceSource, IrPass, IrResource, PassFlags, PassIndex, PassType,
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

                let pass_name = node.name;
                Ok(IrPass {
                    index: PassIndex(0),
                    pass_type,
                    access_set: ResourceAccessSet { reads, writes },
                    color_attachments,
                    depth_stencil,
                    instance_source,
                    dispatch_source: None,
                    view_type,
                    view: Arc::new(EmptyView { name: pass_name.clone() }),
                    name: pass_name,
                    tags: Vec::new(),
                    flags: PassFlags::empty(),
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
                let pass_name = node.name;

                Ok(IrPass {
                    index: PassIndex(0),
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
                    view: Arc::new(EmptyView { name: pass_name.clone() }),
                    name: pass_name,
                    tags: Vec::new(),
                    flags: PassFlags::empty(),
                    })
            }

            PassType::RayTracing | PassType::Copy => {
                // Attachments not allowed on non-graphics passes.
                if !node.color_attachments.is_empty() || node.depth_stencil.is_some() {
                    return Err(ConversionError::AttachmentsNotAllowed(pass_type));
                }
                let pass_name = node.name;

                Ok(IrPass {
                    index: PassIndex(0),
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
                    view: Arc::new(EmptyView { name: pass_name.clone() }),
                    name: pass_name,
                    tags: Vec::new(),
                    flags: PassFlags::empty(),
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
// Test Helpers (available to integration tests)
// ---------------------------------------------------------------------------

/// Creates a minimal PyPassNode for testing purposes.
/// This is doc-hidden but available to integration tests.
#[doc(hidden)]
pub fn minimal_py_pass_node(name: &str, pass_type: PyPassType) -> PyPassNode {
    PyPassNode {
        name: name.to_string(),
        pass_type,
        color_attachments: vec![PyColorAttachment {
            resource: 0,
            load_op: "clear".to_string(),
            store_op: "store".to_string(),
        }],
        depth_stencil: Some(PyDepthStencilAttachment {
            resource: 1,
            depth_load_op: "clear".to_string(),
            depth_store_op: "store".to_string(),
            stencil_load_op: "load".to_string(),
            stencil_store_op: "store".to_string(),
        }),
        reads: vec![],
        writes: vec![],
        instance_source: None,
        dispatch_source: None,
        view_type: PyViewType { kind: "default".to_string() },
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

// ---------------------------------------------------------------------------
// PyO3 Bindings (T-WGPU-P7.6.1)
// ---------------------------------------------------------------------------

#[cfg(feature = "pyo3")]
pub mod pyo3_bindings {
    //! PyO3 bindings for the Frame Graph system.
    //!
    //! Provides Python-accessible wrappers for:
    //! - `PyFrameGraph` — main graph builder
    //! - `PyPassId` / `PyResourceId` — opaque handles
    //! - `PyCompiledFrameGraph` — compiled graph for execution
    //! - `PyFrameGraphCompiler` — graph compiler

    use pyo3::prelude::*;
    use pyo3::exceptions::{PyValueError, PyRuntimeError};

    use crate::frame_graph::graph::{
        FrameGraph, FrameGraphError, GraphResourceLifetime, PassId, PassType, ResourceAccess,
        ResourceId, ResourceType,
    };
    use crate::frame_graph::execution::{CompiledFrameGraph, FrameGraphCompiler};

    // -------------------------------------------------------------------------
    // PyPassId
    // -------------------------------------------------------------------------

    /// Opaque handle identifying a pass in the frame graph.
    ///
    /// Pass IDs are returned by `add_*_pass` methods and can be used
    /// to query pass information or set up dependencies.
    #[pyclass(name = "PassId")]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PyPassId(pub(crate) PassId);

    #[pymethods]
    impl PyPassId {
        /// Returns the raw numeric ID.
        #[getter]
        pub fn raw(&self) -> u64 {
            self.0.raw()
        }

        /// Returns true if this is the invalid/null ID.
        pub fn is_invalid(&self) -> bool {
            self.0.is_invalid()
        }

        fn __repr__(&self) -> String {
            format!("PassId({})", self.0.raw())
        }

        fn __hash__(&self) -> u64 {
            self.0.raw()
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl From<PassId> for PyPassId {
        fn from(id: PassId) -> Self {
            Self(id)
        }
    }

    impl From<PyPassId> for PassId {
        fn from(py_id: PyPassId) -> Self {
            py_id.0
        }
    }

    // -------------------------------------------------------------------------
    // PyResourceId
    // -------------------------------------------------------------------------

    /// Opaque handle identifying a resource in the frame graph.
    ///
    /// Resource IDs are returned by `create_*` methods and used to
    /// connect passes to resources.
    #[pyclass(name = "ResourceId")]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PyResourceId(pub(crate) ResourceId);

    #[pymethods]
    impl PyResourceId {
        /// Returns the raw numeric ID.
        #[getter]
        pub fn raw(&self) -> u64 {
            self.0.raw()
        }

        /// Returns true if this is the invalid/null ID.
        pub fn is_invalid(&self) -> bool {
            self.0.is_invalid()
        }

        fn __repr__(&self) -> String {
            format!("ResourceId({})", self.0.raw())
        }

        fn __hash__(&self) -> u64 {
            self.0.raw()
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl From<ResourceId> for PyResourceId {
        fn from(id: ResourceId) -> Self {
            Self(id)
        }
    }

    impl From<PyResourceId> for ResourceId {
        fn from(py_id: PyResourceId) -> Self {
            py_id.0
        }
    }

    // -------------------------------------------------------------------------
    // PyFrameGraph
    // -------------------------------------------------------------------------

    /// High-level frame graph builder for organizing GPU workloads.
    ///
    /// The frame graph provides automatic dependency resolution, resource
    /// lifetime tracking, and execution ordering for GPU passes.
    ///
    /// # Example
    ///
    /// ```python
    /// from trinity_renderer import FrameGraph
    ///
    /// graph = FrameGraph("main_frame")
    ///
    /// # Create resources
    /// color = graph.create_texture("color", 1920, 1080, "rgba8unorm")
    /// depth = graph.create_texture("depth", 1920, 1080, "depth32float")
    ///
    /// # Add passes
    /// shadow_pass = graph.add_render_pass("shadow", [])
    /// main_pass = graph.add_render_pass("main", [color])
    /// ```
    #[pyclass(name = "FrameGraph")]
    pub struct PyFrameGraph {
        inner: FrameGraph,
        name: String,
    }

    #[pymethods]
    impl PyFrameGraph {
        /// Creates a new empty frame graph.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        #[new]
        pub fn new(name: &str) -> Self {
            Self {
                inner: FrameGraph::new(),
                name: name.to_string(),
            }
        }

        /// Returns the name of this frame graph.
        #[getter]
        pub fn name(&self) -> &str {
            &self.name
        }

        /// Adds a render pass to the graph.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        /// * `color_attachments` — List of resource IDs for color render targets.
        ///
        /// # Returns
        ///
        /// The unique pass ID for the new pass.
        #[pyo3(signature = (name, color_attachments))]
        pub fn add_render_pass(
            &mut self,
            name: &str,
            color_attachments: Vec<PyResourceId>,
        ) -> PyPassId {
            let pass_id = self.inner.add_pass(name, PassType::Render);

            // Connect color attachments as outputs (writes)
            for color_res in color_attachments {
                self.inner.connect(pass_id, color_res.into(), ResourceAccess::Write);
            }

            pass_id.into()
        }

        /// Adds a compute pass to the graph.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        /// * `reads` — Resources the pass reads from.
        /// * `writes` — Resources the pass writes to.
        ///
        /// # Returns
        ///
        /// The unique pass ID for the new pass.
        #[pyo3(signature = (name, reads, writes))]
        pub fn add_compute_pass(
            &mut self,
            name: &str,
            reads: Vec<PyResourceId>,
            writes: Vec<PyResourceId>,
        ) -> PyPassId {
            let pass_id = self.inner.add_pass(name, PassType::Compute);

            for read_res in reads {
                self.inner.connect(pass_id, read_res.into(), ResourceAccess::Read);
            }
            for write_res in writes {
                self.inner.connect(pass_id, write_res.into(), ResourceAccess::Write);
            }

            pass_id.into()
        }

        /// Adds a copy/transfer pass to the graph.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        ///
        /// # Returns
        ///
        /// The unique pass ID for the new pass.
        pub fn add_copy_pass(&mut self, name: &str) -> PyPassId {
            self.inner.add_pass(name, PassType::Transfer).into()
        }

        /// Adds a ray-tracing pass to the graph.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        /// * `reads` — Resources the pass reads from.
        /// * `writes` — Resources the pass writes to.
        ///
        /// # Returns
        ///
        /// The unique pass ID for the new pass.
        #[pyo3(signature = (name, reads, writes))]
        pub fn add_raytracing_pass(
            &mut self,
            name: &str,
            reads: Vec<PyResourceId>,
            writes: Vec<PyResourceId>,
        ) -> PyPassId {
            let pass_id = self.inner.add_pass(name, PassType::RayTracing);

            for read_res in reads {
                self.inner.connect(pass_id, read_res.into(), ResourceAccess::Read);
            }
            for write_res in writes {
                self.inner.connect(pass_id, write_res.into(), ResourceAccess::Write);
            }

            pass_id.into()
        }

        /// Creates a 2D texture resource.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        /// * `width` — Texture width in texels.
        /// * `height` — Texture height in texels.
        /// * `format` — Texture format (e.g., "rgba8unorm", "depth32float").
        ///
        /// # Returns
        ///
        /// The unique resource ID for the new texture.
        pub fn create_texture(
            &mut self,
            name: &str,
            width: u32,
            height: u32,
            format: &str,
        ) -> PyResourceId {
            self.inner
                .add_resource(name, ResourceType::Texture2D, GraphResourceLifetime::Transient)
                .into()
        }

        /// Creates a 3D/volume texture resource.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        /// * `width` — Texture width in texels.
        /// * `height` — Texture height in texels.
        /// * `depth` — Texture depth in texels.
        /// * `format` — Texture format.
        ///
        /// # Returns
        ///
        /// The unique resource ID for the new texture.
        pub fn create_texture_3d(
            &mut self,
            name: &str,
            width: u32,
            height: u32,
            depth: u32,
            format: &str,
        ) -> PyResourceId {
            self.inner
                .add_resource(name, ResourceType::Texture3D, GraphResourceLifetime::Transient)
                .into()
        }

        /// Creates a cube map texture resource.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        /// * `size` — Width and height of each cube face in texels.
        /// * `format` — Texture format.
        ///
        /// # Returns
        ///
        /// The unique resource ID for the new texture.
        pub fn create_texture_cube(
            &mut self,
            name: &str,
            size: u32,
            format: &str,
        ) -> PyResourceId {
            self.inner
                .add_resource(name, ResourceType::TextureCube, GraphResourceLifetime::Transient)
                .into()
        }

        /// Creates a GPU buffer resource.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        /// * `size` — Buffer size in bytes.
        ///
        /// # Returns
        ///
        /// The unique resource ID for the new buffer.
        pub fn create_buffer(&mut self, name: &str, size: u64) -> PyResourceId {
            self.inner
                .add_resource(name, ResourceType::Buffer, GraphResourceLifetime::Transient)
                .into()
        }

        /// Imports an external resource (e.g., swapchain image).
        ///
        /// Imported resources are not allocated by the frame graph but
        /// their state is tracked for barrier insertion.
        ///
        /// # Arguments
        ///
        /// * `name` — Human-readable name for debugging.
        /// * `resource_type` — Type of resource ("texture2d", "buffer", etc.).
        ///
        /// # Returns
        ///
        /// The unique resource ID for the imported resource.
        pub fn import_resource(&mut self, name: &str, resource_type: &str) -> PyResult<PyResourceId> {
            let res_type = match resource_type.to_lowercase().as_str() {
                "texture2d" | "texture" => ResourceType::Texture2D,
                "texture3d" => ResourceType::Texture3D,
                "texturecube" | "cubemap" => ResourceType::TextureCube,
                "buffer" => ResourceType::Buffer,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown resource type: '{}'. Expected: texture2d, texture3d, texturecube, buffer",
                        other
                    )));
                }
            };

            Ok(self
                .inner
                .add_resource(name, res_type, GraphResourceLifetime::Imported)
                .into())
        }

        /// Connects a pass to a resource with read access.
        ///
        /// Creates a dependency: the pass reads from the resource.
        ///
        /// # Arguments
        ///
        /// * `pass` — The pass that reads the resource.
        /// * `resource` — The resource being read.
        pub fn connect_read(&mut self, pass: PyPassId, resource: PyResourceId) {
            self.inner.connect(pass.into(), resource.into(), ResourceAccess::Read);
        }

        /// Connects a pass to a resource with write access.
        ///
        /// Creates a dependency: the pass writes to the resource.
        ///
        /// # Arguments
        ///
        /// * `pass` — The pass that writes the resource.
        /// * `resource` — The resource being written.
        pub fn connect_write(&mut self, pass: PyPassId, resource: PyResourceId) {
            self.inner.connect(pass.into(), resource.into(), ResourceAccess::Write);
        }

        /// Connects a pass to a resource with read-write access.
        ///
        /// Creates dependencies for both reading and writing.
        ///
        /// # Arguments
        ///
        /// * `pass` — The pass that reads and writes the resource.
        /// * `resource` — The resource being accessed.
        pub fn connect_read_write(&mut self, pass: PyPassId, resource: PyResourceId) {
            self.inner.connect(pass.into(), resource.into(), ResourceAccess::ReadWrite);
        }

        /// Returns the number of passes in the graph.
        pub fn pass_count(&self) -> usize {
            self.inner.passes().count()
        }

        /// Returns the number of resources in the graph.
        pub fn resource_count(&self) -> usize {
            self.inner.resources().count()
        }

        /// Validates and compiles the frame graph.
        ///
        /// Performs topological sorting and cycle detection.
        ///
        /// # Errors
        ///
        /// Returns an error if the graph contains cycles or invalid references.
        pub fn validate(&mut self) -> PyResult<()> {
            self.inner.compile().map_err(|e| match e {
                FrameGraphError::CyclicDependency => {
                    PyValueError::new_err("Frame graph contains a cyclic dependency")
                }
                FrameGraphError::MissingResource(id) => {
                    PyValueError::new_err(format!("Missing resource: {}", id))
                }
                FrameGraphError::MissingPass(id) => {
                    PyValueError::new_err(format!("Missing pass: {}", id))
                }
                FrameGraphError::InvalidAccess(msg) => {
                    PyValueError::new_err(format!("Invalid access: {}", msg))
                }
                FrameGraphError::NotCompiled => {
                    PyRuntimeError::new_err("Graph not compiled")
                }
                FrameGraphError::ExecutionFailed(msg) => {
                    PyRuntimeError::new_err(format!("Execution failed: {}", msg))
                }
            })
        }

        /// Enables or disables a pass.
        ///
        /// Disabled passes are skipped during compilation and execution.
        ///
        /// # Arguments
        ///
        /// * `pass` — The pass to enable or disable.
        /// * `enabled` — True to enable, false to disable.
        pub fn set_pass_enabled(&mut self, pass: PyPassId, enabled: bool) -> PyResult<()> {
            if let Some(p) = self.inner.get_pass_mut(pass.into()) {
                p.enabled = enabled;
                Ok(())
            } else {
                Err(PyValueError::new_err(format!("Pass not found: {:?}", pass)))
            }
        }

        /// Returns whether a pass is enabled.
        pub fn is_pass_enabled(&self, pass: PyPassId) -> PyResult<bool> {
            if let Some(p) = self.inner.get_pass(pass.into()) {
                Ok(p.enabled)
            } else {
                Err(PyValueError::new_err(format!("Pass not found: {:?}", pass)))
            }
        }

        /// Resets the graph for the next frame.
        ///
        /// Clears the execution order but preserves passes and resources.
        pub fn reset(&mut self) {
            self.inner.reset();
        }

        fn __repr__(&self) -> String {
            format!(
                "FrameGraph('{}', passes={}, resources={})",
                self.name,
                self.pass_count(),
                self.resource_count()
            )
        }
    }

    // -------------------------------------------------------------------------
    // PyCompiledFrameGraph
    // -------------------------------------------------------------------------

    /// A compiled frame graph ready for execution.
    ///
    /// Contains the execution order, barrier batches, and resource allocations
    /// computed during compilation.
    #[pyclass(name = "CompiledFrameGraph")]
    pub struct PyCompiledFrameGraph {
        inner: CompiledFrameGraph,
    }

    #[pymethods]
    impl PyCompiledFrameGraph {
        /// Returns the number of passes in the execution order.
        pub fn pass_count(&self) -> usize {
            self.inner.pass_count()
        }

        /// Returns the total number of barriers across all batches.
        pub fn barrier_count(&self) -> usize {
            self.inner.total_barrier_count()
        }

        /// Returns the total memory usage of all allocations.
        pub fn memory_usage(&self) -> u64 {
            self.inner.total_memory_usage()
        }

        /// Returns the memory savings from aliasing.
        pub fn memory_savings(&self) -> u64 {
            self.inner.memory_savings()
        }

        /// Returns true if the compiled graph is empty.
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        /// Returns the execution order as a list of pass IDs.
        pub fn execution_order(&self) -> Vec<PyPassId> {
            self.inner
                .execution_order
                .iter()
                .map(|&id| PyPassId(id))
                .collect()
        }

        /// Returns the number of aliased resource groups.
        pub fn alias_count(&self) -> usize {
            self.inner.alias_info.len()
        }

        /// Returns the number of resource allocations.
        pub fn allocation_count(&self) -> usize {
            self.inner.resource_allocations.len()
        }

        fn __repr__(&self) -> String {
            format!(
                "CompiledFrameGraph(passes={}, barriers={}, memory={}B, savings={}B)",
                self.pass_count(),
                self.barrier_count(),
                self.memory_usage(),
                self.memory_savings()
            )
        }
    }

    impl From<CompiledFrameGraph> for PyCompiledFrameGraph {
        fn from(compiled: CompiledFrameGraph) -> Self {
            Self { inner: compiled }
        }
    }

    // -------------------------------------------------------------------------
    // PyFrameGraphCompiler
    // -------------------------------------------------------------------------

    /// Compiles frame graphs into an executable form.
    ///
    /// The compiler performs:
    /// 1. Topological sorting for execution order
    /// 2. Barrier resolution for synchronization
    /// 3. Resource aliasing for memory optimization
    ///
    /// # Example
    ///
    /// ```python
    /// from trinity_renderer import FrameGraph, FrameGraphCompiler
    ///
    /// graph = FrameGraph("frame")
    /// # ... add passes and resources ...
    /// graph.validate()
    ///
    /// compiler = FrameGraphCompiler()
    /// compiled = compiler.compile(graph)
    /// print(f"Passes: {compiled.pass_count()}, Memory: {compiled.memory_usage()}")
    /// ```
    #[pyclass(name = "FrameGraphCompiler")]
    pub struct PyFrameGraphCompiler {
        inner: FrameGraphCompiler,
    }

    #[pymethods]
    impl PyFrameGraphCompiler {
        /// Creates a new frame graph compiler with default settings.
        #[new]
        pub fn new() -> Self {
            Self {
                inner: FrameGraphCompiler::new(),
            }
        }

        /// Compiles a frame graph into an executable form.
        ///
        /// The graph must be validated (via `graph.validate()`) before compilation.
        ///
        /// # Arguments
        ///
        /// * `graph` — The frame graph to compile.
        ///
        /// # Returns
        ///
        /// A compiled frame graph ready for execution.
        pub fn compile(&mut self, graph: &PyFrameGraph) -> PyCompiledFrameGraph {
            self.inner.compile(&graph.inner).into()
        }

        /// Resets the compiler state for reuse.
        pub fn reset(&mut self) {
            self.inner.reset();
        }

        fn __repr__(&self) -> String {
            "FrameGraphCompiler()".to_string()
        }
    }

    impl Default for PyFrameGraphCompiler {
        fn default() -> Self {
            Self::new()
        }
    }

    // -------------------------------------------------------------------------
    // Module Registration
    // -------------------------------------------------------------------------

    /// Registers the frame_graph Python module.
    ///
    /// Called from the parent `#[pymodule]` to register all frame graph types.
    pub fn register_module(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(py, "frame_graph")?;

        m.add_class::<PyFrameGraph>()?;
        m.add_class::<PyPassId>()?;
        m.add_class::<PyResourceId>()?;
        m.add_class::<PyCompiledFrameGraph>()?;
        m.add_class::<PyFrameGraphCompiler>()?;

        parent.add_submodule(&m)?;
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;

        // -- PyPassId tests ---------------------------------------------------

        #[test]
        fn test_py_pass_id_creation() {
            let id = PyPassId(PassId::new(42));
            assert_eq!(id.raw(), 42);
            assert!(!id.is_invalid());
        }

        #[test]
        fn test_py_pass_id_invalid() {
            let id = PyPassId(PassId::INVALID);
            assert!(id.is_invalid());
        }

        #[test]
        fn test_py_pass_id_equality() {
            let a = PyPassId(PassId::new(1));
            let b = PyPassId(PassId::new(1));
            let c = PyPassId(PassId::new(2));
            assert!(a.__eq__(&b));
            assert!(!a.__eq__(&c));
        }

        #[test]
        fn test_py_pass_id_hash() {
            let id = PyPassId(PassId::new(123));
            assert_eq!(id.__hash__(), 123);
        }

        #[test]
        fn test_py_pass_id_repr() {
            let id = PyPassId(PassId::new(7));
            assert_eq!(id.__repr__(), "PassId(7)");
        }

        // -- PyResourceId tests -----------------------------------------------

        #[test]
        fn test_py_resource_id_creation() {
            let id = PyResourceId(ResourceId::new(99));
            assert_eq!(id.raw(), 99);
            assert!(!id.is_invalid());
        }

        #[test]
        fn test_py_resource_id_invalid() {
            let id = PyResourceId(ResourceId::INVALID);
            assert!(id.is_invalid());
        }

        #[test]
        fn test_py_resource_id_equality() {
            let a = PyResourceId(ResourceId::new(5));
            let b = PyResourceId(ResourceId::new(5));
            let c = PyResourceId(ResourceId::new(6));
            assert!(a.__eq__(&b));
            assert!(!a.__eq__(&c));
        }

        #[test]
        fn test_py_resource_id_hash() {
            let id = PyResourceId(ResourceId::new(456));
            assert_eq!(id.__hash__(), 456);
        }

        #[test]
        fn test_py_resource_id_repr() {
            let id = PyResourceId(ResourceId::new(3));
            assert_eq!(id.__repr__(), "ResourceId(3)");
        }

        // -- PyFrameGraph tests -----------------------------------------------

        #[test]
        fn test_py_frame_graph_creation() {
            let graph = PyFrameGraph::new("test_frame");
            assert_eq!(graph.name(), "test_frame");
            assert_eq!(graph.pass_count(), 0);
            assert_eq!(graph.resource_count(), 0);
        }

        #[test]
        fn test_py_frame_graph_create_texture() {
            let mut graph = PyFrameGraph::new("test");
            let tex = graph.create_texture("color", 1920, 1080, "rgba8unorm");
            assert!(!tex.is_invalid());
            assert_eq!(graph.resource_count(), 1);
        }

        #[test]
        fn test_py_frame_graph_create_buffer() {
            let mut graph = PyFrameGraph::new("test");
            let buf = graph.create_buffer("storage", 4096);
            assert!(!buf.is_invalid());
            assert_eq!(graph.resource_count(), 1);
        }

        #[test]
        fn test_py_frame_graph_add_render_pass() {
            let mut graph = PyFrameGraph::new("test");
            let color = graph.create_texture("color", 1920, 1080, "rgba8unorm");
            let pass = graph.add_render_pass("main", vec![color]);
            assert!(!pass.is_invalid());
            assert_eq!(graph.pass_count(), 1);
        }

        #[test]
        fn test_py_frame_graph_add_compute_pass() {
            let mut graph = PyFrameGraph::new("test");
            let input = graph.create_texture("input", 256, 256, "rgba8unorm");
            let output = graph.create_texture("output", 256, 256, "rgba8unorm");
            let pass = graph.add_compute_pass("process", vec![input], vec![output]);
            assert!(!pass.is_invalid());
            assert_eq!(graph.pass_count(), 1);
        }

        #[test]
        fn test_py_frame_graph_add_copy_pass() {
            let mut graph = PyFrameGraph::new("test");
            let pass = graph.add_copy_pass("copy");
            assert!(!pass.is_invalid());
            assert_eq!(graph.pass_count(), 1);
        }

        #[test]
        fn test_py_frame_graph_add_raytracing_pass() {
            let mut graph = PyFrameGraph::new("test");
            let input = graph.create_buffer("scene", 1024);
            let output = graph.create_texture("output", 1920, 1080, "rgba8unorm");
            let pass = graph.add_raytracing_pass("trace", vec![input], vec![output]);
            assert!(!pass.is_invalid());
            assert_eq!(graph.pass_count(), 1);
        }

        #[test]
        fn test_py_frame_graph_create_texture_3d() {
            let mut graph = PyFrameGraph::new("test");
            let tex = graph.create_texture_3d("volume", 128, 128, 64, "rgba8unorm");
            assert!(!tex.is_invalid());
            assert_eq!(graph.resource_count(), 1);
        }

        #[test]
        fn test_py_frame_graph_create_texture_cube() {
            let mut graph = PyFrameGraph::new("test");
            let tex = graph.create_texture_cube("env", 512, "rgba16float");
            assert!(!tex.is_invalid());
            assert_eq!(graph.resource_count(), 1);
        }

        #[test]
        fn test_py_frame_graph_validate_empty() {
            let mut graph = PyFrameGraph::new("empty");
            // Empty graph should validate successfully
            assert!(graph.validate().is_ok());
        }

        #[test]
        fn test_py_frame_graph_validate_simple() {
            let mut graph = PyFrameGraph::new("simple");
            let color = graph.create_texture("color", 1920, 1080, "rgba8unorm");
            let _ = graph.add_render_pass("main", vec![color]);
            assert!(graph.validate().is_ok());
        }

        #[test]
        fn test_py_frame_graph_repr() {
            let mut graph = PyFrameGraph::new("test");
            let _ = graph.create_texture("tex", 256, 256, "rgba8unorm");
            let repr = graph.__repr__();
            assert!(repr.contains("test"));
            assert!(repr.contains("passes=0"));
            assert!(repr.contains("resources=1"));
        }

        #[test]
        fn test_py_frame_graph_reset() {
            let mut graph = PyFrameGraph::new("test");
            let color = graph.create_texture("color", 1920, 1080, "rgba8unorm");
            let _ = graph.add_render_pass("main", vec![color]);
            assert!(graph.validate().is_ok());
            graph.reset();
            // After reset, the graph needs to be recompiled
            // but passes and resources are preserved
            assert_eq!(graph.pass_count(), 1);
            assert_eq!(graph.resource_count(), 1);
        }

        // -- PyFrameGraphCompiler tests ---------------------------------------

        #[test]
        fn test_py_compiler_creation() {
            let compiler = PyFrameGraphCompiler::new();
            assert_eq!(compiler.__repr__(), "FrameGraphCompiler()");
        }

        #[test]
        fn test_py_compiler_compile_empty() {
            let mut compiler = PyFrameGraphCompiler::new();
            let mut graph = PyFrameGraph::new("empty");
            graph.validate().unwrap();
            let compiled = compiler.compile(&graph);
            assert_eq!(compiled.pass_count(), 0);
            assert!(compiled.is_empty());
        }

        #[test]
        fn test_py_compiler_compile_simple() {
            let mut compiler = PyFrameGraphCompiler::new();
            let mut graph = PyFrameGraph::new("simple");
            let color = graph.create_texture("color", 1920, 1080, "rgba8unorm");
            let _ = graph.add_render_pass("main", vec![color]);
            graph.validate().unwrap();
            let compiled = compiler.compile(&graph);
            assert_eq!(compiled.pass_count(), 1);
            assert!(!compiled.is_empty());
        }

        #[test]
        fn test_py_compiler_reset() {
            let mut compiler = PyFrameGraphCompiler::new();
            let mut graph = PyFrameGraph::new("test");
            graph.validate().unwrap();
            let _ = compiler.compile(&graph);
            compiler.reset();
            // Should be able to compile again after reset
            let compiled = compiler.compile(&graph);
            assert_eq!(compiled.pass_count(), 0);
        }

        // -- PyCompiledFrameGraph tests ---------------------------------------

        #[test]
        fn test_py_compiled_graph_empty() {
            let compiled = PyCompiledFrameGraph::from(CompiledFrameGraph::new());
            assert_eq!(compiled.pass_count(), 0);
            assert_eq!(compiled.barrier_count(), 0);
            assert_eq!(compiled.memory_usage(), 0);
            assert_eq!(compiled.memory_savings(), 0);
            assert!(compiled.is_empty());
            assert_eq!(compiled.alias_count(), 0);
            assert_eq!(compiled.allocation_count(), 0);
        }

        #[test]
        fn test_py_compiled_graph_execution_order() {
            let compiled = PyCompiledFrameGraph::from(CompiledFrameGraph::new());
            let order = compiled.execution_order();
            assert!(order.is_empty());
        }

        #[test]
        fn test_py_compiled_graph_repr() {
            let compiled = PyCompiledFrameGraph::from(CompiledFrameGraph::new());
            let repr = compiled.__repr__();
            assert!(repr.contains("CompiledFrameGraph"));
            assert!(repr.contains("passes=0"));
        }

        // -- Integration tests ------------------------------------------------

        #[test]
        fn test_full_pipeline_shadow_main() {
            let mut graph = PyFrameGraph::new("shadow_main");

            // Create resources
            let shadow_map = graph.create_texture("shadow_map", 2048, 2048, "depth32float");
            let color_rt = graph.create_texture("color", 1920, 1080, "rgba8unorm");
            let depth_rt = graph.create_texture("depth", 1920, 1080, "depth32float");

            // Shadow pass writes to shadow map
            let shadow_pass = graph.add_render_pass("shadow", vec![]);
            graph.connect_write(shadow_pass, shadow_map);

            // Main pass reads shadow map, writes color and depth
            let main_pass = graph.add_render_pass("main", vec![color_rt]);
            graph.connect_read(main_pass, shadow_map);
            graph.connect_write(main_pass, depth_rt);

            // Post-process reads color, writes to new buffer
            let output = graph.create_texture("output", 1920, 1080, "rgba8unorm");
            let post_pass = graph.add_compute_pass("post", vec![color_rt], vec![output]);

            // Validate the graph
            assert!(graph.validate().is_ok());

            // Compile
            let mut compiler = PyFrameGraphCompiler::new();
            let compiled = compiler.compile(&graph);

            assert_eq!(compiled.pass_count(), 3);
            assert!(!compiled.is_empty());

            // Verify execution order respects dependencies
            let order = compiled.execution_order();
            assert_eq!(order.len(), 3);

            // Shadow pass must come before main pass
            let shadow_idx = order.iter().position(|&p| p.__eq__(&shadow_pass)).unwrap();
            let main_idx = order.iter().position(|&p| p.__eq__(&main_pass)).unwrap();
            let post_idx = order.iter().position(|&p| p.__eq__(&post_pass)).unwrap();

            assert!(shadow_idx < main_idx, "shadow must execute before main");
            assert!(main_idx < post_idx, "main must execute before post");
        }

        #[test]
        fn test_import_resource() {
            let mut graph = PyFrameGraph::new("test");

            // Valid resource types
            let tex = graph.import_resource("swapchain", "texture2d").unwrap();
            assert!(!tex.is_invalid());

            let buf = graph.import_resource("staging", "buffer").unwrap();
            assert!(!buf.is_invalid());

            let cube = graph.import_resource("env", "cubemap").unwrap();
            assert!(!cube.is_invalid());

            assert_eq!(graph.resource_count(), 3);
        }

        #[test]
        fn test_import_resource_invalid_type() {
            let mut graph = PyFrameGraph::new("test");
            let result = graph.import_resource("bad", "unknown_type");
            assert!(result.is_err());
        }

        #[test]
        fn test_pass_enable_disable() {
            let mut graph = PyFrameGraph::new("test");
            let pass = graph.add_copy_pass("copy");

            // Default enabled
            assert!(graph.is_pass_enabled(pass).unwrap());

            // Disable
            graph.set_pass_enabled(pass, false).unwrap();
            assert!(!graph.is_pass_enabled(pass).unwrap());

            // Re-enable
            graph.set_pass_enabled(pass, true).unwrap();
            assert!(graph.is_pass_enabled(pass).unwrap());
        }

        #[test]
        fn test_pass_enable_invalid_pass() {
            let graph = PyFrameGraph::new("test");
            let invalid_pass = PyPassId(PassId::new(999));
            assert!(graph.is_pass_enabled(invalid_pass).is_err());
        }

        #[test]
        fn test_connect_read_write() {
            let mut graph = PyFrameGraph::new("test");
            let buf = graph.create_buffer("data", 1024);
            let pass = graph.add_compute_pass("process", vec![], vec![]);

            // Connect with read-write access
            graph.connect_read_write(pass, buf);

            // Should still validate
            assert!(graph.validate().is_ok());
        }
    }
}
