//! WGPU barrier command generation (Phase 4c of the frame graph compiler).
//!
//! Provides bitflag types mirroring `wgpu::TextureUsages` and
//! `wgpu::BufferUsages`, a mapping from TRINITY [`ResourceState`] to those
//! flags, and functions to resolve compiler-internal barrier descriptors into
//! wgpu-style barrier commands.
//!
//! The types in this module are self-defined (no wgpu crate dependency) so the
//! runtime backend can pass the bitflag values directly to the wgpu-native FFI
//! layer without linking against the Rust `wgpu` crate.

use core::fmt;
use core::ops::BitOr;

use crate::frame_graph::{
    BarrierCommand, BarrierDescriptor, BarrierTuple, EdgeType, IrResource, PassIndex,
    ResourceDesc, ResourceHandle, ResourceState, resource_desc_is_texture,
};

// ---------------------------------------------------------------------------
// WgpuTextureUsage — bitflags mirroring wgpu::TextureUsages
// ---------------------------------------------------------------------------

/// Bitflags for `wgpu::TextureUsages` (self-defined, no wgpu crate dependency).
///
/// Values mirror the wgpu-native bitflag constants so the runtime can pass
/// them directly to the FFI layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WgpuTextureUsage(u32);

#[allow(dead_code)]
impl WgpuTextureUsage {
    pub const COPY_SRC: Self = Self(1);
    pub const COPY_DST: Self = Self(2);
    pub const TEXTURE_BINDING: Self = Self(4);
    pub const STORAGE_BINDING: Self = Self(8);
    pub const RENDER_ATTACHMENT: Self = Self(16);
    pub const PRESENT: Self = Self(32);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn bits(&self) -> u32 {
        self.0
    }

    pub fn contains(&self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl BitOr for WgpuTextureUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl fmt::Display for WgpuTextureUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<&str> = Vec::new();
        if self.contains(Self::COPY_SRC) {
            parts.push("COPY_SRC");
        }
        if self.contains(Self::COPY_DST) {
            parts.push("COPY_DST");
        }
        if self.contains(Self::TEXTURE_BINDING) {
            parts.push("TEXTURE_BINDING");
        }
        if self.contains(Self::STORAGE_BINDING) {
            parts.push("STORAGE_BINDING");
        }
        if self.contains(Self::RENDER_ATTACHMENT) {
            parts.push("RENDER_ATTACHMENT");
        }
        if self.contains(Self::PRESENT) {
            parts.push("PRESENT");
        }
        if parts.is_empty() {
            write!(f, "(empty)")
        } else {
            write!(f, "{}", parts.join(" | "))
        }
    }
}

// ---------------------------------------------------------------------------
// WgpuBufferUsage — bitflags mirroring wgpu::BufferUsages
// ---------------------------------------------------------------------------

/// Bitflags for `wgpu::BufferUsages` (self-defined, no wgpu crate dependency).
///
/// Values mirror the wgpu-native bitflag constants so the runtime can pass
/// them directly to the FFI layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WgpuBufferUsage(u32);

#[allow(dead_code)]
impl WgpuBufferUsage {
    pub const COPY_SRC: Self = Self(1);
    pub const COPY_DST: Self = Self(2);
    pub const INDEX: Self = Self(4);
    pub const VERTEX: Self = Self(8);
    pub const UNIFORM: Self = Self(16);
    pub const STORAGE: Self = Self(32);
    pub const INDIRECT: Self = Self(64);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn bits(&self) -> u32 {
        self.0
    }

    pub fn contains(&self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl BitOr for WgpuBufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl fmt::Display for WgpuBufferUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<&str> = Vec::new();
        if self.contains(Self::COPY_SRC) {
            parts.push("COPY_SRC");
        }
        if self.contains(Self::COPY_DST) {
            parts.push("COPY_DST");
        }
        if self.contains(Self::INDEX) {
            parts.push("INDEX");
        }
        if self.contains(Self::VERTEX) {
            parts.push("VERTEX");
        }
        if self.contains(Self::UNIFORM) {
            parts.push("UNIFORM");
        }
        if self.contains(Self::STORAGE) {
            parts.push("STORAGE");
        }
        if self.contains(Self::INDIRECT) {
            parts.push("INDIRECT");
        }
        if parts.is_empty() {
            write!(f, "(empty)")
        } else {
            write!(f, "{}", parts.join(" | "))
        }
    }
}

// ---------------------------------------------------------------------------
// ResourceState → wgpu usage mapping
// ---------------------------------------------------------------------------

/// Maps a TRINITY [`ResourceState`] to the equivalent texture-usage bitflags
/// and buffer-usage bitflags required by wgpu.
///
/// The first element of the returned tuple is the texture usage (zero if the
/// state has no texture counterpart). The second element is the buffer usage
/// (zero if the state has no buffer counterpart).
pub fn resource_state_to_wgpu_usage(
    state: ResourceState,
) -> (WgpuTextureUsage, WgpuBufferUsage) {
    use crate::frame_graph::ResourceState::*;
    match state {
        Uninitialized => (WgpuTextureUsage::empty(), WgpuBufferUsage::empty()),

        // Buffer-only states.
        VertexBuffer => (WgpuTextureUsage::empty(), WgpuBufferUsage::VERTEX),
        IndexBuffer => (WgpuTextureUsage::empty(), WgpuBufferUsage::INDEX),
        IndirectArgument => (WgpuTextureUsage::empty(), WgpuBufferUsage::INDIRECT),

        // Texture-only states.
        ColorAttachment | DepthStencilAttachment => {
            (WgpuTextureUsage::RENDER_ATTACHMENT, WgpuBufferUsage::empty())
        }
        DepthStencilReadOnly => {
            (WgpuTextureUsage::TEXTURE_BINDING, WgpuBufferUsage::empty())
        }

        // Shared (both texture and buffer).
        ShaderRead => (
            WgpuTextureUsage::TEXTURE_BINDING,
            WgpuBufferUsage::UNIFORM,
        ),
        ShaderReadWrite => (
            WgpuTextureUsage::STORAGE_BINDING,
            WgpuBufferUsage::STORAGE,
        ),
        TransferSrc => (WgpuTextureUsage::COPY_SRC, WgpuBufferUsage::COPY_SRC),
        TransferDst => (WgpuTextureUsage::COPY_DST, WgpuBufferUsage::COPY_DST),

        // Specialised.
        AccelerationStructure => (
            WgpuTextureUsage::STORAGE_BINDING,
            WgpuBufferUsage::STORAGE,
        ),
        Present => (WgpuTextureUsage::PRESENT, WgpuBufferUsage::empty()),
    }
}

// ---------------------------------------------------------------------------
// WgpuBarrier — resolved barrier command
// ---------------------------------------------------------------------------

/// A resolved GPU barrier command targeting a wgpu-style resource.
///
/// Each variant carries the wgpu usage bitflags derived from the TRINITY
/// resource-state transition, plus sub-resource (texture) or byte-range
/// (buffer) selectors.  The runtime backend consumes this type directly
/// when encoding `wgpu::CommandEncoder::resource_barrier()` calls.
#[derive(Clone, Debug, PartialEq)]
pub enum WgpuBarrier {
    /// Barrier that transitions a texture resource between usage states.
    Texture {
        /// Logical handle of the resource being transitioned.
        resource: ResourceHandle,
        /// Bitflags representing the texture usage before the transition.
        from: WgpuTextureUsage,
        /// Bitflags representing the texture usage after the transition.
        to: WgpuTextureUsage,
        /// Range of mip levels affected (`None` = all).
        mip_levels: Option<std::ops::Range<u32>>,
        /// Range of array layers affected (`None` = all).
        array_layers: Option<std::ops::Range<u32>>,
    },
    /// Barrier that transitions a buffer resource between usage states.
    Buffer {
        /// Logical handle of the resource being transitioned.
        resource: ResourceHandle,
        /// Bitflags representing the buffer usage before the transition.
        from: WgpuBufferUsage,
        /// Bitflags representing the buffer usage after the transition.
        to: WgpuBufferUsage,
        /// Byte offset of the affected range (`None` = entire buffer).
        offset: Option<u64>,
        /// Size in bytes of the affected range (`None` = entire buffer).
        size: Option<u64>,
    },
}

impl WgpuBarrier {
    /// Returns the [`ResourceHandle`] targeted by this barrier.
    pub fn resource(&self) -> ResourceHandle {
        match self {
            WgpuBarrier::Texture { resource, .. }
            | WgpuBarrier::Buffer { resource, .. } => *resource,
        }
    }
}

impl fmt::Display for WgpuBarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WgpuBarrier::Texture {
                resource,
                from,
                to,
                mip_levels,
                array_layers,
            } => {
                write!(
                    f,
                    "WgpuBarrier::Texture(res={}, from={}, to={}",
                    resource, from, to,
                )?;
                if let Some(mips) = mip_levels {
                    write!(f, ", mips={}..{}", mips.start, mips.end)?;
                }
                if let Some(layers) = array_layers {
                    write!(f, ", layers={}..{}", layers.start, layers.end)?;
                }
                write!(f, ")")
            }
            WgpuBarrier::Buffer {
                resource,
                from,
                to,
                offset,
                size,
            } => {
                write!(
                    f,
                    "WgpuBarrier::Buffer(res={}, from={}, to={}",
                    resource, from, to,
                )?;
                if let Some(off) = offset {
                    write!(f, ", offset={}", off)?;
                }
                if let Some(sz) = size {
                    write!(f, ", size={}", sz)?;
                }
                write!(f, ")")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// generate_wgpu_barriers — from 6-element compiler tuples
// ---------------------------------------------------------------------------

/// Generates wgpu-style barrier commands from the compiler-internal barrier
/// representation.
///
/// Accepts the 6-element tuple format produced by Phase 4 barrier scheduling:
/// `(from_pass, to_pass, resource_handle, edge_type, before_state, after_state)`.
///
/// Resources are resolved via the `resources` slice to determine whether each
/// transition targets a texture or a buffer.
pub fn generate_wgpu_barriers(
    barriers: &[(
        PassIndex,
        PassIndex,
        ResourceHandle,
        EdgeType,
        ResourceState,
        ResourceState,
    )],
    resources: &[IrResource],
) -> Vec<WgpuBarrier> {
    use std::collections::HashMap;

    // Build descriptor lookup.
    let desc_map: HashMap<ResourceHandle, &ResourceDesc> =
        resources.iter().map(|r| (r.handle, &r.desc)).collect();

    let mut wgpu_barriers: Vec<WgpuBarrier> = Vec::with_capacity(barriers.len());

    for &(_from, _to, handle, _edge, before, after) in barriers {
        let desc = match desc_map.get(&handle) {
            Some(d) => *d,
            None => continue,
        };

        if resource_desc_is_texture(desc) {
            let (from_usage, _) = resource_state_to_wgpu_usage(before);
            let (to_usage, _) = resource_state_to_wgpu_usage(after);
            wgpu_barriers.push(WgpuBarrier::Texture {
                resource: handle,
                from: from_usage,
                to: to_usage,
                mip_levels: None,
                array_layers: None,
            });
        } else {
            let (_, from_usage) = resource_state_to_wgpu_usage(before);
            let (_, to_usage) = resource_state_to_wgpu_usage(after);
            wgpu_barriers.push(WgpuBarrier::Buffer {
                resource: handle,
                from: from_usage,
                to: to_usage,
                offset: None,
                size: None,
            });
        }
    }

    wgpu_barriers
}

// ---------------------------------------------------------------------------
// WgpuBarrierResolveContext — ergonomic resolution from TRINITY descriptors
// ---------------------------------------------------------------------------

/// Context for resolving wgpu barriers from the frame-graph compiler output.
///
/// Wraps the compiled frame graph's resource table to provide ergonomic
/// resolution of logical resource handles to their physical type, which
/// determines whether a barrier should be emitted as a texture or buffer
/// transition.
pub struct WgpuBarrierResolveContext<'a> {
    resources: &'a [IrResource],
}

impl<'a> WgpuBarrierResolveContext<'a> {
    /// Creates a new resolve context from the compiled frame graph's resources.
    pub fn new(resources: &'a [IrResource]) -> Self {
        Self { resources }
    }

    /// Determines whether the resource identified by `handle` is a texture.
    ///
    /// Returns `None` when the handle is not found in the resource table.
    pub fn is_texture(&self, handle: ResourceHandle) -> Option<bool> {
        self.resources
            .iter()
            .find(|r| r.handle == handle)
            .map(|r| resource_desc_is_texture(&r.desc))
    }

    /// Resolves a TRINITY barrier descriptor into a `WgpuBarrier`.
    ///
    /// Returns `None` if the resource handle does not exist in the context's
    /// resource table.
    pub fn resolve(&self, barrier: &BarrierDescriptor) -> Option<WgpuBarrier> {
        let handle = barrier.resource();
        let _desc = self.resources.iter().find(|r| r.handle == handle)?;

        match barrier {
            BarrierDescriptor::Texture(td) => {
                let (from, _) = resource_state_to_wgpu_usage(td.before);
                let (to, _) = resource_state_to_wgpu_usage(td.after);
                Some(WgpuBarrier::Texture {
                    resource: handle,
                    from,
                    to,
                    mip_levels: td.mip_levels.clone(),
                    array_layers: td.array_layers.clone(),
                })
            }
            BarrierDescriptor::Buffer(bd) => {
                let (_, from) = resource_state_to_wgpu_usage(bd.before);
                let (_, to) = resource_state_to_wgpu_usage(bd.after);
                Some(WgpuBarrier::Buffer {
                    resource: handle,
                    from,
                    to,
                    offset: bd.offset,
                    size: bd.size,
                })
            }
        }
    }

    /// Resolves a single [`BarrierTuple`] (the 6-tuple compiler format) into
    /// a [`WgpuBarrier`].
    ///
    /// The 6-tuple fields are `(from_pass, to_pass, resource_handle, edge_type,
    /// before_state, after_state)`.  Only `resource_handle`, `before_state`,
    /// and `after_state` are used for the barrier — pass indices and edge type
    /// are metadata consumed by scheduling, not by the barrier itself.
    ///
    /// Returns `None` if the resource handle cannot be found in the context's
    /// resource table.
    pub fn resolve_barrier_tuple(&self, bt: &BarrierTuple) -> Option<WgpuBarrier> {
        let (_from, _to, handle, _edge, before, after) = *bt;
        let _desc = self.resources.iter().find(|r| r.handle == handle)?;

        if self.is_texture(handle)? {
            let (from_tex, _) = resource_state_to_wgpu_usage(before);
            let (to_tex, _) = resource_state_to_wgpu_usage(after);
            Some(WgpuBarrier::Texture {
                resource: handle,
                from: from_tex,
                to: to_tex,
                mip_levels: None,
                array_layers: None,
            })
        } else {
            let (_, from_buf) = resource_state_to_wgpu_usage(before);
            let (_, to_buf) = resource_state_to_wgpu_usage(after);
            Some(WgpuBarrier::Buffer {
                resource: handle,
                from: from_buf,
                to: to_buf,
                offset: None,
                size: None,
            })
        }
    }

    /// Resolves a batch of [`BarrierTuple`] records into a `Vec<WgpuBarrier>`.
    ///
    /// This is the primary entry point for translating `ScheduledPass::pre_barriers`
    /// or `ScheduledPass::post_barriers` into wgpu-compatible barrier commands.
    ///
    /// Barriers whose resource handle is not found in the context are silently
    /// skipped (this should not happen in a well-formed compiled graph).
    pub fn resolve_barrier_tuples(&self, barriers: &[BarrierTuple]) -> Vec<WgpuBarrier> {
        let mut out: Vec<WgpuBarrier> = Vec::with_capacity(barriers.len());
        for bt in barriers {
            if let Some(wb) = self.resolve_barrier_tuple(bt) {
                out.push(wb);
            }
        }
        out
    }

    /// Resolves an entire `BarrierCommand` batch into a `Vec<WgpuBarrier>`.
    ///
    /// Barriers whose resource handle is not found in the context are silently
    /// skipped (this should not happen in a well-formed compiled graph).
    pub fn resolve_batch(&self, cmd: &BarrierCommand) -> Vec<WgpuBarrier> {
        let mut out: Vec<WgpuBarrier> = Vec::with_capacity(
            cmd.texture_barriers.len() + cmd.buffer_barriers.len(),
        );

        for td in &cmd.texture_barriers {
            if let Some(wb) = self.resolve(&BarrierDescriptor::Texture(td.clone())) {
                out.push(wb);
            }
        }

        for bd in &cmd.buffer_barriers {
            if let Some(wb) = self.resolve(&BarrierDescriptor::Buffer(bd.clone())) {
                out.push(wb);
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use crate::frame_graph::{
        BarrierTuple, BufferDesc, EdgeType, IrResource, PassIndex, ResourceDesc,
        ResourceHandle, ResourceLifetime, ResourceState, TextureDesc,
    };

    use super::*;

    // -- Bitflag constant values ----------------------------------------------

    #[test]
    fn test_wgpu_texture_usage_bitflag_values() {
        assert_eq!(WgpuTextureUsage::COPY_SRC.bits(), 1);
        assert_eq!(WgpuTextureUsage::COPY_DST.bits(), 2);
        assert_eq!(WgpuTextureUsage::TEXTURE_BINDING.bits(), 4);
        assert_eq!(WgpuTextureUsage::STORAGE_BINDING.bits(), 8);
        assert_eq!(WgpuTextureUsage::RENDER_ATTACHMENT.bits(), 16);
        assert_eq!(WgpuTextureUsage::PRESENT.bits(), 32);
    }

    #[test]
    fn test_wgpu_buffer_usage_bitflag_values() {
        assert_eq!(WgpuBufferUsage::COPY_SRC.bits(), 1);
        assert_eq!(WgpuBufferUsage::COPY_DST.bits(), 2);
        assert_eq!(WgpuBufferUsage::INDEX.bits(), 4);
        assert_eq!(WgpuBufferUsage::VERTEX.bits(), 8);
        assert_eq!(WgpuBufferUsage::UNIFORM.bits(), 16);
        assert_eq!(WgpuBufferUsage::STORAGE.bits(), 32);
        assert_eq!(WgpuBufferUsage::INDIRECT.bits(), 64);
    }

    // -- resource_state_to_wgpu_usage mapping ---------------------------------

    #[test]
    fn test_resource_state_shader_read() {
        let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::ShaderRead);
        assert_eq!(tex, WgpuTextureUsage::TEXTURE_BINDING);
        assert_eq!(buf, WgpuBufferUsage::UNIFORM);
    }

    #[test]
    fn test_resource_state_render_target() {
        let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::ColorAttachment);
        assert_eq!(tex, WgpuTextureUsage::RENDER_ATTACHMENT);
        assert_eq!(buf, WgpuBufferUsage::empty());
    }

    #[test]
    fn test_resource_state_depth_stencil() {
        let (tex, buf) =
            resource_state_to_wgpu_usage(ResourceState::DepthStencilAttachment);
        assert_eq!(tex, WgpuTextureUsage::RENDER_ATTACHMENT);
        assert_eq!(buf, WgpuBufferUsage::empty());
    }

    #[test]
    fn test_resource_state_present() {
        let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::Present);
        assert_eq!(tex, WgpuTextureUsage::PRESENT);
        assert_eq!(buf, WgpuBufferUsage::empty());
    }

    #[test]
    fn test_resource_state_unordered_access() {
        let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::ShaderReadWrite);
        assert_eq!(tex, WgpuTextureUsage::STORAGE_BINDING);
        assert_eq!(buf, WgpuBufferUsage::STORAGE);
    }

    #[test]
    fn test_resource_state_copy_src() {
        let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::TransferSrc);
        assert_eq!(tex, WgpuTextureUsage::COPY_SRC);
        assert_eq!(buf, WgpuBufferUsage::COPY_SRC);
    }

    #[test]
    fn test_resource_state_copy_dst() {
        let (tex, buf) = resource_state_to_wgpu_usage(ResourceState::TransferDst);
        assert_eq!(tex, WgpuTextureUsage::COPY_DST);
        assert_eq!(buf, WgpuBufferUsage::COPY_DST);
    }

    // -- generate_wgpu_barriers -----------------------------------------------

    #[test]
    fn test_generate_wgpu_barriers_texture() {
        let res = IrResource::new(
            ResourceHandle(1),
            "color_rt",
            ResourceDesc::Texture2D(TextureDesc {
                width: 256,
                height: 256,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        );

        let barrier_tuple = (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ColorAttachment,
        );

        let result = generate_wgpu_barriers(&[barrier_tuple], &[res]);
        assert_eq!(result.len(), 1);
        assert!(
            matches!(result[0], WgpuBarrier::Texture { .. }),
            "expected a Texture barrier"
        );
        if let WgpuBarrier::Texture {
            resource,
            ref from,
            ref to,
            ..
        } = result[0]
        {
            assert_eq!(resource, ResourceHandle(1));
            assert_eq!(*from, WgpuTextureUsage::empty());
            assert_eq!(*to, WgpuTextureUsage::RENDER_ATTACHMENT);
        }
    }

    #[test]
    fn test_generate_wgpu_barriers_buffer() {
        let res = IrResource::new(
            ResourceHandle(2),
            "storage_buf",
            ResourceDesc::Buffer(BufferDesc {
                size: 4096,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        );

        let barrier_tuple = (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ShaderReadWrite,
        );

        let result = generate_wgpu_barriers(&[barrier_tuple], &[res]);
        assert_eq!(result.len(), 1);
        assert!(
            matches!(result[0], WgpuBarrier::Buffer { .. }),
            "expected a Buffer barrier"
        );
        if let WgpuBarrier::Buffer {
            resource,
            ref from,
            ref to,
            ..
        } = result[0]
        {
            assert_eq!(resource, ResourceHandle(2));
            assert_eq!(*from, WgpuBufferUsage::empty());
            assert_eq!(*to, WgpuBufferUsage::STORAGE);
        }
    }

    #[test]
    fn test_generate_wgpu_barriers_empty() {
        let result = generate_wgpu_barriers(&[], &[]);
        assert!(result.is_empty());
    }

    // -- WgpuBarrierResolveContext --------------------------------------------

    #[test]
    fn test_resolve_context_is_texture() {
        let resources = vec![
            IrResource::new(
                ResourceHandle(1),
                "tex",
                ResourceDesc::Texture2D(TextureDesc {
                    width: 256,
                    height: 256,
                    mip_levels: 1,
                    array_layers: 1,
                    format: "rgba8unorm".into(),
                }),
                ResourceLifetime::Transient,
                ResourceState::Uninitialized,
            ),
            IrResource::new(
                ResourceHandle(2),
                "buf",
                ResourceDesc::Buffer(BufferDesc {
                    size: 4096,
                    usage: "storage".into(),
                    is_indirect_arg: false,
                }),
                ResourceLifetime::Transient,
                ResourceState::Uninitialized,
            ),
        ];

        let ctx = WgpuBarrierResolveContext::new(&resources);
        assert_eq!(ctx.is_texture(ResourceHandle(1)), Some(true));
        assert_eq!(ctx.is_texture(ResourceHandle(2)), Some(false));
        assert_eq!(ctx.is_texture(ResourceHandle(99)), None);
    }

    // -- resolve_barrier_tuple --------------------------------------------------

    #[test]
    fn test_resolve_barrier_tuple_texture() {
        let resources = vec![IrResource::new(
            ResourceHandle(1),
            "color_rt",
            ResourceDesc::Texture2D(TextureDesc {
                width: 256,
                height: 256,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        )];
        let ctx = WgpuBarrierResolveContext::new(&resources);

        let bt: BarrierTuple = (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ColorAttachment,
        );
        let result = ctx.resolve_barrier_tuple(&bt);
        assert!(result.is_some());
        let wb = result.unwrap();
        assert!(matches!(wb, WgpuBarrier::Texture { .. }));
        if let WgpuBarrier::Texture {
            resource, from, to, ..
        } = wb
        {
            assert_eq!(resource, ResourceHandle(1));
            assert_eq!(from, WgpuTextureUsage::empty());
            assert_eq!(to, WgpuTextureUsage::RENDER_ATTACHMENT);
        }
    }

    #[test]
    fn test_resolve_barrier_tuple_buffer() {
        let resources = vec![IrResource::new(
            ResourceHandle(2),
            "storage_buf",
            ResourceDesc::Buffer(BufferDesc {
                size: 4096,
                usage: "storage".into(),
                is_indirect_arg: false,
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        )];
        let ctx = WgpuBarrierResolveContext::new(&resources);

        let bt: BarrierTuple = (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(2),
            EdgeType::RAW,
            ResourceState::ShaderReadWrite,
            ResourceState::TransferSrc,
        );
        let result = ctx.resolve_barrier_tuple(&bt);
        assert!(result.is_some());
        let wb = result.unwrap();
        assert!(matches!(wb, WgpuBarrier::Buffer { .. }));
        if let WgpuBarrier::Buffer {
            resource, from, to, ..
        } = wb
        {
            assert_eq!(resource, ResourceHandle(2));
            assert_eq!(from, WgpuBufferUsage::STORAGE);
            assert_eq!(to, WgpuBufferUsage::COPY_SRC);
        }
    }

    #[test]
    fn test_resolve_barrier_tuple_unknown_handle_returns_none() {
        let resources = vec![IrResource::new(
            ResourceHandle(1),
            "tex",
            ResourceDesc::Texture2D(TextureDesc {
                width: 256,
                height: 256,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        )];
        let ctx = WgpuBarrierResolveContext::new(&resources);

        // Handle 99 does not exist in the context.
        let bt: BarrierTuple = (
            PassIndex(3),
            PassIndex(4),
            ResourceHandle(99),
            EdgeType::RAW,
            ResourceState::Uninitialized,
            ResourceState::ColorAttachment,
        );
        assert!(ctx.resolve_barrier_tuple(&bt).is_none());
    }

    // -- resolve_barrier_tuples (batch) -----------------------------------------

    #[test]
    fn test_resolve_barrier_tuples_mixed_types() {
        let resources = vec![
            IrResource::new(
                ResourceHandle(1),
                "tex",
                ResourceDesc::Texture2D(TextureDesc {
                    width: 256,
                    height: 256,
                    mip_levels: 1,
                    array_layers: 1,
                    format: "rgba8unorm".into(),
                }),
                ResourceLifetime::Transient,
                ResourceState::Uninitialized,
            ),
            IrResource::new(
                ResourceHandle(2),
                "buf",
                ResourceDesc::Buffer(BufferDesc {
                    size: 4096,
                    usage: "storage".into(),
                    is_indirect_arg: false,
                }),
                ResourceLifetime::Transient,
                ResourceState::Uninitialized,
            ),
        ];
        let ctx = WgpuBarrierResolveContext::new(&resources);

        let bts: Vec<BarrierTuple> = vec![
            (
                PassIndex(0),
                PassIndex(1),
                ResourceHandle(1),
                EdgeType::RAW,
                ResourceState::Uninitialized,
                ResourceState::ColorAttachment,
            ),
            (
                PassIndex(0),
                PassIndex(1),
                ResourceHandle(2),
                EdgeType::RAW,
                ResourceState::Uninitialized,
                ResourceState::ShaderReadWrite,
            ),
        ];

        let result = ctx.resolve_barrier_tuples(&bts);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0], WgpuBarrier::Texture { .. }));
        assert!(matches!(result[1], WgpuBarrier::Buffer { .. }));
    }

    #[test]
    fn test_resolve_barrier_tuples_empty() {
        let ctx = WgpuBarrierResolveContext::new(&[]);
        let result = ctx.resolve_barrier_tuples(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_barrier_tuples_unknown_handle_skipped() {
        let resources = vec![IrResource::new(
            ResourceHandle(1),
            "tex",
            ResourceDesc::Texture2D(TextureDesc {
                width: 256,
                height: 256,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        )];
        let ctx = WgpuBarrierResolveContext::new(&resources);

        let bts: Vec<BarrierTuple> = vec![
            (
                PassIndex(0),
                PassIndex(1),
                ResourceHandle(1),
                EdgeType::RAW,
                ResourceState::Uninitialized,
                ResourceState::ColorAttachment,
            ),
            (
                PassIndex(0),
                PassIndex(1),
                ResourceHandle(99),
                EdgeType::RAW,
                ResourceState::ShaderReadWrite,
                ResourceState::ShaderRead,
            ),
        ];

        let result = ctx.resolve_barrier_tuples(&bts);
        // Only the known handle produces a barrier; the unknown one is skipped.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].resource(), ResourceHandle(1));
    }

    #[test]
    fn test_resolve_barrier_tuples_with_display() {
        let resources = vec![IrResource::new(
            ResourceHandle(1),
            "albedo",
            ResourceDesc::Texture2D(TextureDesc {
                width: 256,
                height: 256,
                mip_levels: 1,
                array_layers: 1,
                format: "rgba8unorm".into(),
            }),
            ResourceLifetime::Transient,
            ResourceState::Uninitialized,
        )];
        let ctx = WgpuBarrierResolveContext::new(&resources);

        let bt: BarrierTuple = (
            PassIndex(0),
            PassIndex(1),
            ResourceHandle(1),
            EdgeType::RAW,
            ResourceState::ColorAttachment,
            ResourceState::ShaderRead,
        );
        let wb = ctx.resolve_barrier_tuple(&bt).unwrap();
        let display_str = format!("{}", wb);
        assert!(display_str.contains("albedo") || display_str.contains("res="));
        assert!(display_str.contains("RENDER_ATTACHMENT") || display_str.contains("TEXTURE_BINDING"));
    }
}
