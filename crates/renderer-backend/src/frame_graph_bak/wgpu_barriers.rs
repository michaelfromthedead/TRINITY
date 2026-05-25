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

use crate::frame_graph::{
    BarrierCommand, BarrierDescriptor, EdgeType, IrResource, PassIndex, ResourceDesc,
    ResourceHandle, ResourceState, resource_desc_is_texture,
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
