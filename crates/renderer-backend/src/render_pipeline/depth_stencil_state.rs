//! Depth and stencil state configuration for wgpu 25.x render pipelines.
//!
//! This module provides depth test and stencil state abstractions with common presets,
//! builder patterns, and comprehensive metadata for tooling and debugging.
//!
//! # Overview
//!
//! The depth-stencil state controls two key GPU pipeline operations:
//!
//! 1. **Depth Testing**: Determines if fragments are visible based on their Z-depth
//! 2. **Stencil Testing**: Per-pixel masking using an integer stencil buffer
//!
//! # Depth Test Configuration (T-WGPU-P3.6.1)
//!
//! Depth testing compares fragment depth against the depth buffer:
//!
//! | Compare Function | Passes When | Use Case |
//! |------------------|-------------|----------|
//! | `Never` | Never passes | Discard all fragments |
//! | `Less` | fragment < buffer | Standard forward rendering |
//! | `Equal` | fragment == buffer | Decals, exact depth match |
//! | `LessEqual` | fragment <= buffer | Common default |
//! | `Greater` | fragment > buffer | Reverse-Z rendering |
//! | `NotEqual` | fragment != buffer | Outline effects |
//! | `GreaterEqual` | fragment >= buffer | Reverse-Z with equal |
//! | `Always` | Always passes | Disable depth test |
//!
//! # Stencil State Configuration (T-WGPU-P3.6.2)
//!
//! Stencil operations modify the stencil buffer based on test results:
//!
//! | Operation | Effect | Use Case |
//! |-----------|--------|----------|
//! | `Keep` | No change | Pass-through |
//! | `Zero` | Set to 0 | Clear stencil |
//! | `Replace` | Set to reference | Write mask value |
//! | `IncrementClamp` | Increment, clamp at max | Shadow volumes |
//! | `DecrementClamp` | Decrement, clamp at 0 | Shadow volumes |
//! | `Invert` | Bitwise invert | Toggle mask |
//! | `IncrementWrap` | Increment, wrap to 0 | Counting |
//! | `DecrementWrap` | Decrement, wrap to max | Counting |
//!
//! # Depth Formats
//!
//! | Format | Bits | Stencil | Use Case |
//! |--------|------|---------|----------|
//! | `Depth32Float` | 32 depth | No | High precision, no stencil |
//! | `Depth24Plus` | 24+ depth | No | Standard depth-only |
//! | `Depth24PlusStencil8` | 24 depth + 8 stencil | Yes | Depth + stencil |
//! | `Depth32FloatStencil8` | 32 depth + 8 stencil | Yes | Maximum precision |
//!
//! # wgpu API Reference
//!
//! ```ignore
//! pub struct DepthStencilState {
//!     pub format: TextureFormat,
//!     pub depth_write_enabled: bool,
//!     pub depth_compare: CompareFunction,
//!     pub stencil: StencilState,
//!     pub bias: DepthBiasState,
//! }
//!
//! pub struct StencilState {
//!     pub front: StencilFaceState,
//!     pub back: StencilFaceState,
//!     pub read_mask: u32,
//!     pub write_mask: u32,
//! }
//!
//! pub struct StencilFaceState {
//!     pub compare: CompareFunction,
//!     pub fail_op: StencilOperation,
//!     pub depth_fail_op: StencilOperation,
//!     pub pass_op: StencilOperation,
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::depth_stencil_state::{
//!     DepthStencilStateDescriptor, DepthStencilStateBuilder
//! };
//!
//! // Use a preset
//! let depth_less = DepthStencilStateDescriptor::depth_less();
//! let reverse_z = DepthStencilStateDescriptor::depth_greater();
//!
//! // Custom configuration
//! let custom = DepthStencilStateBuilder::new()
//!     .format(wgpu::TextureFormat::Depth24PlusStencil8)
//!     .depth_write_enabled(true)
//!     .depth_compare(wgpu::CompareFunction::LessEqual)
//!     .stencil_read_mask(0xFF)
//!     .stencil_write_mask(0xFF)
//!     .build();
//!
//! // Convert to wgpu type
//! let wgpu_state: wgpu::DepthStencilState = depth_less.into();
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// DepthStencilStateDescriptor
// ---------------------------------------------------------------------------

/// Describes depth and stencil testing state for a render pipeline.
///
/// # Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `format` | `TextureFormat` | Depth/stencil texture format |
/// | `depth_write_enabled` | `bool` | Whether to write depth values |
/// | `depth_compare` | `CompareFunction` | Depth comparison function |
/// | `stencil_front` | `StencilFaceStateDescriptor` | Front face stencil state |
/// | `stencil_back` | `StencilFaceStateDescriptor` | Back face stencil state |
/// | `stencil_read_mask` | `u32` | Stencil read mask |
/// | `stencil_write_mask` | `u32` | Stencil write mask |
/// | `bias` | `DepthBiasStateDescriptor` | Depth bias configuration |
///
/// # Defaults
///
/// - `format`: `Depth24PlusStencil8`
/// - `depth_write_enabled`: `true`
/// - `depth_compare`: `Less`
/// - `stencil_front/back`: All operations `Keep`, compare `Always`
/// - `stencil_read_mask`: `0xFFFFFFFF`
/// - `stencil_write_mask`: `0xFFFFFFFF`
/// - `bias`: No bias (all zeros)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStencilStateDescriptor {
    /// The texture format of the depth/stencil attachment.
    pub format: wgpu::TextureFormat,
    /// Whether depth values should be written to the depth buffer.
    pub depth_write_enabled: bool,
    /// The comparison function for depth testing.
    pub depth_compare: wgpu::CompareFunction,
    /// Stencil state for front-facing primitives.
    pub stencil_front: StencilFaceStateDescriptor,
    /// Stencil state for back-facing primitives.
    pub stencil_back: StencilFaceStateDescriptor,
    /// Bitmask applied when reading from the stencil buffer.
    pub stencil_read_mask: u32,
    /// Bitmask applied when writing to the stencil buffer.
    pub stencil_write_mask: u32,
    /// Depth bias configuration for polygon offset.
    pub bias: DepthBiasStateDescriptor,
}

impl Default for DepthStencilStateDescriptor {
    fn default() -> Self {
        Self {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil_front: StencilFaceStateDescriptor::default(),
            stencil_back: StencilFaceStateDescriptor::default(),
            stencil_read_mask: 0xFFFFFFFF,
            stencil_write_mask: 0xFFFFFFFF,
            bias: DepthBiasStateDescriptor::default(),
        }
    }
}

impl DepthStencilStateDescriptor {
    /// Create a new depth-stencil state with default values.
    ///
    /// Defaults to `Depth24PlusStencil8` format with `Less` comparison
    /// and depth writes enabled.
    pub fn new() -> Self {
        Self::default()
    }

    // -------------------------------------------------------------------------
    // Depth Presets (T-WGPU-P3.6.1)
    // -------------------------------------------------------------------------

    /// Depth test with `Less` comparison and writes enabled.
    ///
    /// Standard forward rendering configuration where closer fragments
    /// (smaller depth values) overwrite farther ones.
    ///
    /// # Configuration
    /// - `depth_write_enabled`: `true`
    /// - `depth_compare`: `Less`
    /// - `format`: `Depth24PlusStencil8`
    pub fn depth_less() -> Self {
        Self {
            depth_compare: wgpu::CompareFunction::Less,
            depth_write_enabled: true,
            ..Default::default()
        }
    }

    /// Depth test with `LessEqual` comparison and writes enabled.
    ///
    /// Common default that allows fragments at exactly the same depth
    /// to pass. Useful when rendering coplanar geometry.
    ///
    /// # Configuration
    /// - `depth_write_enabled`: `true`
    /// - `depth_compare`: `LessEqual`
    pub fn depth_less_equal() -> Self {
        Self {
            depth_compare: wgpu::CompareFunction::LessEqual,
            depth_write_enabled: true,
            ..Default::default()
        }
    }

    /// Depth test with `Always` comparison - effectively disabled.
    ///
    /// All fragments pass depth test. Useful for:
    /// - Full-screen effects
    /// - UI rendering
    /// - Post-processing passes
    ///
    /// # Configuration
    /// - `depth_write_enabled`: `false`
    /// - `depth_compare`: `Always`
    pub fn depth_always() -> Self {
        Self {
            depth_compare: wgpu::CompareFunction::Always,
            depth_write_enabled: false,
            ..Default::default()
        }
    }

    /// Depth test with `Greater` comparison for reverse-Z rendering.
    ///
    /// Reverse-Z provides better depth precision distribution by mapping
    /// near plane to 1.0 and far plane to 0.0. Requires `Greater` comparison.
    ///
    /// # Benefits of Reverse-Z
    /// - Better precision near the camera
    /// - Reduced z-fighting at distance
    /// - More uniform precision distribution
    ///
    /// # Configuration
    /// - `depth_write_enabled`: `true`
    /// - `depth_compare`: `Greater`
    pub fn depth_greater() -> Self {
        Self {
            depth_compare: wgpu::CompareFunction::Greater,
            depth_write_enabled: true,
            ..Default::default()
        }
    }

    /// Depth test with `GreaterEqual` comparison for reverse-Z.
    ///
    /// Like `depth_greater()` but allows equal depth values to pass.
    ///
    /// # Configuration
    /// - `depth_write_enabled`: `true`
    /// - `depth_compare`: `GreaterEqual`
    pub fn depth_greater_equal() -> Self {
        Self {
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            depth_write_enabled: true,
            ..Default::default()
        }
    }

    /// Depth test with `Never` comparison - all fragments fail.
    ///
    /// No fragments will pass depth test. Useful for:
    /// - Stencil-only passes
    /// - Debug visualization
    ///
    /// # Configuration
    /// - `depth_write_enabled`: `false`
    /// - `depth_compare`: `Never`
    pub fn depth_never() -> Self {
        Self {
            depth_compare: wgpu::CompareFunction::Never,
            depth_write_enabled: false,
            ..Default::default()
        }
    }

    /// Read-only depth test with `Less` comparison.
    ///
    /// Tests against depth buffer but doesn't write. Useful for:
    /// - Transparent object rendering
    /// - Decal projection
    /// - Soft particles
    ///
    /// # Configuration
    /// - `depth_write_enabled`: `false`
    /// - `depth_compare`: `Less`
    pub fn depth_read_only() -> Self {
        Self {
            depth_compare: wgpu::CompareFunction::Less,
            depth_write_enabled: false,
            ..Default::default()
        }
    }

    /// Read-only depth test with `Greater` for reverse-Z.
    ///
    /// Tests against depth buffer (reverse-Z) but doesn't write.
    ///
    /// # Configuration
    /// - `depth_write_enabled`: `false`
    /// - `depth_compare`: `Greater`
    pub fn depth_read_only_reverse_z() -> Self {
        Self {
            depth_compare: wgpu::CompareFunction::Greater,
            depth_write_enabled: false,
            ..Default::default()
        }
    }

    /// Configuration for transparent object rendering.
    ///
    /// Tests depth but doesn't write, allowing proper transparent sorting.
    ///
    /// # Configuration
    /// - `depth_write_enabled`: `false`
    /// - `depth_compare`: `LessEqual`
    pub fn transparent() -> Self {
        Self {
            depth_compare: wgpu::CompareFunction::LessEqual,
            depth_write_enabled: false,
            ..Default::default()
        }
    }

    /// Configuration for shadow map generation.
    ///
    /// High-precision depth-only format with bias for shadow acne prevention.
    ///
    /// # Configuration
    /// - `format`: `Depth32Float`
    /// - `depth_write_enabled`: `true`
    /// - `depth_compare`: `Less`
    /// - `bias`: Shadow map preset (constant: 2, slope: 2.0)
    pub fn shadow_map() -> Self {
        Self {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            bias: DepthBiasStateDescriptor::shadow_map(),
            stencil_front: StencilFaceStateDescriptor::default(),
            stencil_back: StencilFaceStateDescriptor::default(),
            stencil_read_mask: 0,
            stencil_write_mask: 0,
        }
    }

    /// Configuration for depth pre-pass.
    ///
    /// Renders depth only (no color) for early-Z optimization.
    ///
    /// # Configuration
    /// - `format`: `Depth32Float`
    /// - `depth_write_enabled`: `true`
    /// - `depth_compare`: `Less`
    pub fn depth_prepass() -> Self {
        Self {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            ..Default::default()
        }
    }

    // -------------------------------------------------------------------------
    // Stencil Presets (T-WGPU-P3.6.2)
    // -------------------------------------------------------------------------

    /// Stencil write preset - writes reference value on pass.
    ///
    /// Used to mark regions in the stencil buffer for later masking.
    ///
    /// # Configuration
    /// - Depth test: `Always` (disabled)
    /// - Stencil compare: `Always`
    /// - Stencil pass_op: `Replace`
    pub fn stencil_write() -> Self {
        let stencil = StencilFaceStateDescriptor {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Replace,
        };
        Self {
            depth_compare: wgpu::CompareFunction::Always,
            depth_write_enabled: false,
            stencil_front: stencil,
            stencil_back: stencil,
            ..Default::default()
        }
    }

    /// Stencil replace preset - always replaces stencil value.
    ///
    /// # Configuration
    /// - Stencil compare: `Always`
    /// - All operations: `Replace`
    pub fn stencil_replace() -> Self {
        let stencil = StencilFaceStateDescriptor {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Replace,
            depth_fail_op: wgpu::StencilOperation::Replace,
            pass_op: wgpu::StencilOperation::Replace,
        };
        Self {
            depth_compare: wgpu::CompareFunction::Always,
            depth_write_enabled: false,
            stencil_front: stencil,
            stencil_back: stencil,
            ..Default::default()
        }
    }

    /// Stencil increment preset - increments on pass.
    ///
    /// Used for shadow volume counting or marking multiple overlaps.
    ///
    /// # Configuration
    /// - Stencil compare: `Always`
    /// - pass_op: `IncrementClamp`
    pub fn stencil_increment() -> Self {
        let stencil = StencilFaceStateDescriptor {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::IncrementClamp,
        };
        Self {
            depth_compare: wgpu::CompareFunction::Always,
            depth_write_enabled: false,
            stencil_front: stencil,
            stencil_back: stencil,
            ..Default::default()
        }
    }

    /// Stencil decrement preset - decrements on pass.
    ///
    /// Counterpart to increment for shadow volume exit faces.
    ///
    /// # Configuration
    /// - Stencil compare: `Always`
    /// - pass_op: `DecrementClamp`
    pub fn stencil_decrement() -> Self {
        let stencil = StencilFaceStateDescriptor {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::DecrementClamp,
        };
        Self {
            depth_compare: wgpu::CompareFunction::Always,
            depth_write_enabled: false,
            stencil_front: stencil,
            stencil_back: stencil,
            ..Default::default()
        }
    }

    /// Stencil invert preset - inverts stencil bits on pass.
    ///
    /// Useful for XOR-style masking operations.
    ///
    /// # Configuration
    /// - Stencil compare: `Always`
    /// - pass_op: `Invert`
    pub fn stencil_invert() -> Self {
        let stencil = StencilFaceStateDescriptor {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Invert,
        };
        Self {
            depth_compare: wgpu::CompareFunction::Always,
            depth_write_enabled: false,
            stencil_front: stencil,
            stencil_back: stencil,
            ..Default::default()
        }
    }

    /// Stencil zero preset - clears stencil on pass.
    ///
    /// # Configuration
    /// - Stencil compare: `Always`
    /// - pass_op: `Zero`
    pub fn stencil_zero() -> Self {
        let stencil = StencilFaceStateDescriptor {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Zero,
        };
        Self {
            depth_compare: wgpu::CompareFunction::Always,
            depth_write_enabled: false,
            stencil_front: stencil,
            stencil_back: stencil,
            ..Default::default()
        }
    }

    /// Stencil read with `Equal` comparison.
    ///
    /// Only renders where stencil equals reference value.
    /// Useful for masked rendering after stencil write pass.
    ///
    /// # Configuration
    /// - Stencil compare: `Equal`
    /// - All operations: `Keep`
    pub fn stencil_read_equal() -> Self {
        let stencil = StencilFaceStateDescriptor {
            compare: wgpu::CompareFunction::Equal,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Keep,
        };
        Self {
            depth_compare: wgpu::CompareFunction::LessEqual,
            depth_write_enabled: true,
            stencil_front: stencil,
            stencil_back: stencil,
            ..Default::default()
        }
    }

    /// Stencil read with `NotEqual` comparison.
    ///
    /// Only renders where stencil differs from reference value.
    /// Useful for rendering outside a masked region.
    ///
    /// # Configuration
    /// - Stencil compare: `NotEqual`
    /// - All operations: `Keep`
    pub fn stencil_read_not_equal() -> Self {
        let stencil = StencilFaceStateDescriptor {
            compare: wgpu::CompareFunction::NotEqual,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Keep,
        };
        Self {
            depth_compare: wgpu::CompareFunction::LessEqual,
            depth_write_enabled: true,
            stencil_front: stencil,
            stencil_back: stencil,
            ..Default::default()
        }
    }

    // -------------------------------------------------------------------------
    // Fluent API
    // -------------------------------------------------------------------------

    /// Set the depth/stencil texture format.
    pub fn format(mut self, format: wgpu::TextureFormat) -> Self {
        self.format = format;
        self
    }

    /// Enable or disable depth writing.
    pub fn depth_write_enabled(mut self, enabled: bool) -> Self {
        self.depth_write_enabled = enabled;
        self
    }

    /// Set the depth comparison function.
    pub fn depth_compare(mut self, compare: wgpu::CompareFunction) -> Self {
        self.depth_compare = compare;
        self
    }

    /// Set the front face stencil state.
    pub fn stencil_front(mut self, state: StencilFaceStateDescriptor) -> Self {
        self.stencil_front = state;
        self
    }

    /// Set the back face stencil state.
    pub fn stencil_back(mut self, state: StencilFaceStateDescriptor) -> Self {
        self.stencil_back = state;
        self
    }

    /// Set both front and back stencil states to the same configuration.
    pub fn stencil_both(mut self, state: StencilFaceStateDescriptor) -> Self {
        self.stencil_front = state;
        self.stencil_back = state;
        self
    }

    /// Set the stencil read mask.
    pub fn stencil_read_mask(mut self, mask: u32) -> Self {
        self.stencil_read_mask = mask;
        self
    }

    /// Set the stencil write mask.
    pub fn stencil_write_mask(mut self, mask: u32) -> Self {
        self.stencil_write_mask = mask;
        self
    }

    /// Set the depth bias configuration.
    pub fn bias(mut self, bias: DepthBiasStateDescriptor) -> Self {
        self.bias = bias;
        self
    }

    // -------------------------------------------------------------------------
    // Query Methods
    // -------------------------------------------------------------------------

    /// Check if depth testing is effectively enabled.
    ///
    /// Returns false if compare is `Always` (all pass) and writes disabled.
    pub fn is_depth_test_enabled(&self) -> bool {
        self.depth_compare != wgpu::CompareFunction::Always || self.depth_write_enabled
    }

    /// Check if stencil testing is effectively enabled.
    ///
    /// Returns true if either face has a non-trivial configuration.
    pub fn is_stencil_test_enabled(&self) -> bool {
        !self.stencil_front.is_disabled() || !self.stencil_back.is_disabled()
    }

    /// Check if the format includes a stencil component.
    pub fn has_stencil_format(&self) -> bool {
        has_stencil(self.format)
    }

    /// Check if the format is a depth format.
    pub fn is_depth_format(&self) -> bool {
        is_depth_format(self.format)
    }

    /// Check if depth bias is active.
    pub fn has_depth_bias(&self) -> bool {
        self.bias.is_active()
    }
}

// Thread-safety: DepthStencilStateDescriptor contains only Copy types
unsafe impl Send for DepthStencilStateDescriptor {}
unsafe impl Sync for DepthStencilStateDescriptor {}

impl From<DepthStencilStateDescriptor> for wgpu::DepthStencilState {
    fn from(desc: DepthStencilStateDescriptor) -> Self {
        wgpu::DepthStencilState {
            format: desc.format,
            depth_write_enabled: desc.depth_write_enabled,
            depth_compare: desc.depth_compare,
            stencil: wgpu::StencilState {
                front: desc.stencil_front.into(),
                back: desc.stencil_back.into(),
                read_mask: desc.stencil_read_mask,
                write_mask: desc.stencil_write_mask,
            },
            bias: desc.bias.into(),
        }
    }
}

impl From<wgpu::DepthStencilState> for DepthStencilStateDescriptor {
    fn from(state: wgpu::DepthStencilState) -> Self {
        Self {
            format: state.format,
            depth_write_enabled: state.depth_write_enabled,
            depth_compare: state.depth_compare,
            stencil_front: state.stencil.front.into(),
            stencil_back: state.stencil.back.into(),
            stencil_read_mask: state.stencil.read_mask,
            stencil_write_mask: state.stencil.write_mask,
            bias: state.bias.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// StencilFaceStateDescriptor
// ---------------------------------------------------------------------------

/// Describes stencil operations for one face (front or back).
///
/// # Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `compare` | `CompareFunction` | Stencil comparison function |
/// | `fail_op` | `StencilOperation` | Operation when stencil test fails |
/// | `depth_fail_op` | `StencilOperation` | Operation when depth test fails |
/// | `pass_op` | `StencilOperation` | Operation when both tests pass |
///
/// # Stencil Test Flow
///
/// ```text
/// 1. Apply read_mask to stencil buffer value
/// 2. Compare with reference using `compare` function
/// 3. If stencil fails -> apply `fail_op`
/// 4. If stencil passes but depth fails -> apply `depth_fail_op`
/// 5. If both pass -> apply `pass_op`
/// ```
///
/// # Defaults
///
/// Default is a pass-through configuration:
/// - `compare`: `Always`
/// - All operations: `Keep`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFaceStateDescriptor {
    /// Comparison function for the stencil test.
    pub compare: wgpu::CompareFunction,
    /// Operation when stencil test fails.
    pub fail_op: wgpu::StencilOperation,
    /// Operation when stencil passes but depth test fails.
    pub depth_fail_op: wgpu::StencilOperation,
    /// Operation when both stencil and depth tests pass.
    pub pass_op: wgpu::StencilOperation,
}

impl Default for StencilFaceStateDescriptor {
    fn default() -> Self {
        Self {
            compare: wgpu::CompareFunction::Always,
            fail_op: wgpu::StencilOperation::Keep,
            depth_fail_op: wgpu::StencilOperation::Keep,
            pass_op: wgpu::StencilOperation::Keep,
        }
    }
}

impl StencilFaceStateDescriptor {
    /// Create a new stencil face state with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a disabled stencil face state (all Keep, compare Always).
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Set the comparison function.
    pub fn compare(mut self, compare: wgpu::CompareFunction) -> Self {
        self.compare = compare;
        self
    }

    /// Set the fail operation.
    pub fn fail_op(mut self, op: wgpu::StencilOperation) -> Self {
        self.fail_op = op;
        self
    }

    /// Set the depth fail operation.
    pub fn depth_fail_op(mut self, op: wgpu::StencilOperation) -> Self {
        self.depth_fail_op = op;
        self
    }

    /// Set the pass operation.
    pub fn pass_op(mut self, op: wgpu::StencilOperation) -> Self {
        self.pass_op = op;
        self
    }

    /// Set all operations to the same value.
    pub fn all_ops(mut self, op: wgpu::StencilOperation) -> Self {
        self.fail_op = op;
        self.depth_fail_op = op;
        self.pass_op = op;
        self
    }

    /// Check if this face state is effectively disabled.
    ///
    /// A face is disabled if compare is Always and all ops are Keep.
    pub fn is_disabled(&self) -> bool {
        self.compare == wgpu::CompareFunction::Always
            && self.fail_op == wgpu::StencilOperation::Keep
            && self.depth_fail_op == wgpu::StencilOperation::Keep
            && self.pass_op == wgpu::StencilOperation::Keep
    }

    /// Check if any operation modifies the stencil buffer.
    pub fn modifies_stencil(&self) -> bool {
        self.fail_op != wgpu::StencilOperation::Keep
            || self.depth_fail_op != wgpu::StencilOperation::Keep
            || self.pass_op != wgpu::StencilOperation::Keep
    }
}

impl From<StencilFaceStateDescriptor> for wgpu::StencilFaceState {
    fn from(desc: StencilFaceStateDescriptor) -> Self {
        wgpu::StencilFaceState {
            compare: desc.compare,
            fail_op: desc.fail_op,
            depth_fail_op: desc.depth_fail_op,
            pass_op: desc.pass_op,
        }
    }
}

impl From<wgpu::StencilFaceState> for StencilFaceStateDescriptor {
    fn from(state: wgpu::StencilFaceState) -> Self {
        Self {
            compare: state.compare,
            fail_op: state.fail_op,
            depth_fail_op: state.depth_fail_op,
            pass_op: state.pass_op,
        }
    }
}

// ---------------------------------------------------------------------------
// DepthBiasStateDescriptor
// ---------------------------------------------------------------------------

/// Describes depth bias (polygon offset) for depth-stencil state.
///
/// # Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `constant` | `i32` | Fixed depth offset in depth buffer units |
/// | `slope_scale` | `f32` | Slope-dependent depth offset |
/// | `clamp` | `f32` | Maximum absolute bias (0 = no clamp) |
///
/// # Depth Bias Calculation
///
/// ```text
/// bias = constant * r + slope_scale * max_slope
/// final_depth = fragment_depth + clamp(bias, -clamp, clamp)
/// ```
///
/// Where `r` is the minimum resolvable depth difference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthBiasStateDescriptor {
    /// Constant depth value added to each fragment.
    pub constant: i32,
    /// Slope-scaled depth bias.
    pub slope_scale: f32,
    /// Maximum depth bias clamp value.
    pub clamp: f32,
}

impl Default for DepthBiasStateDescriptor {
    fn default() -> Self {
        Self {
            constant: 0,
            slope_scale: 0.0,
            clamp: 0.0,
        }
    }
}

impl DepthBiasStateDescriptor {
    /// Create a new depth bias with default values (no bias).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a depth bias with no offset.
    pub fn none() -> Self {
        Self::default()
    }

    /// Create a preset for shadow map rendering.
    ///
    /// Prevents shadow acne with moderate bias values.
    pub fn shadow_map() -> Self {
        Self {
            constant: 2,
            slope_scale: 2.0,
            clamp: 0.0,
        }
    }

    /// Create a preset for cascaded shadow maps.
    ///
    /// Slightly higher bias for CSM's lower precision at distance.
    pub fn cascaded_shadow_map() -> Self {
        Self {
            constant: 4,
            slope_scale: 3.0,
            clamp: 0.0,
        }
    }

    /// Create a preset for decal rendering.
    pub fn decal() -> Self {
        Self {
            constant: 1,
            slope_scale: 1.0,
            clamp: 0.0,
        }
    }

    /// Create a preset for outline rendering.
    pub fn outline() -> Self {
        Self {
            constant: -1,
            slope_scale: -1.0,
            clamp: 0.0,
        }
    }

    /// Set the constant bias.
    pub fn constant(mut self, constant: i32) -> Self {
        self.constant = constant;
        self
    }

    /// Set the slope scale.
    pub fn slope_scale(mut self, slope_scale: f32) -> Self {
        self.slope_scale = slope_scale;
        self
    }

    /// Set the clamp value.
    pub fn clamp(mut self, clamp: f32) -> Self {
        self.clamp = clamp;
        self
    }

    /// Check if bias is active (non-zero).
    pub fn is_active(&self) -> bool {
        self.constant != 0 || self.slope_scale != 0.0
    }
}

impl From<DepthBiasStateDescriptor> for wgpu::DepthBiasState {
    fn from(desc: DepthBiasStateDescriptor) -> Self {
        wgpu::DepthBiasState {
            constant: desc.constant,
            slope_scale: desc.slope_scale,
            clamp: desc.clamp,
        }
    }
}

impl From<wgpu::DepthBiasState> for DepthBiasStateDescriptor {
    fn from(state: wgpu::DepthBiasState) -> Self {
        Self {
            constant: state.constant,
            slope_scale: state.slope_scale,
            clamp: state.clamp,
        }
    }
}

// ---------------------------------------------------------------------------
// DepthStencilStateBuilder
// ---------------------------------------------------------------------------

/// Builder for creating depth-stencil state configurations.
///
/// # Example
///
/// ```ignore
/// let state = DepthStencilStateBuilder::new()
///     .format(wgpu::TextureFormat::Depth32Float)
///     .depth_write_enabled(true)
///     .depth_compare(wgpu::CompareFunction::LessEqual)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct DepthStencilStateBuilder {
    state: DepthStencilStateDescriptor,
}

impl Default for DepthStencilStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DepthStencilStateBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            state: DepthStencilStateDescriptor::default(),
        }
    }

    /// Start building from a preset.
    pub fn from_preset(preset: DepthStencilStateDescriptor) -> Self {
        Self { state: preset }
    }

    /// Set the depth/stencil texture format.
    pub fn format(mut self, format: wgpu::TextureFormat) -> Self {
        self.state.format = format;
        self
    }

    /// Set depth write enabled.
    pub fn depth_write_enabled(mut self, enabled: bool) -> Self {
        self.state.depth_write_enabled = enabled;
        self
    }

    /// Set the depth comparison function.
    pub fn depth_compare(mut self, compare: wgpu::CompareFunction) -> Self {
        self.state.depth_compare = compare;
        self
    }

    /// Set the front face stencil state.
    pub fn stencil_front(mut self, state: StencilFaceStateDescriptor) -> Self {
        self.state.stencil_front = state;
        self
    }

    /// Set the back face stencil state.
    pub fn stencil_back(mut self, state: StencilFaceStateDescriptor) -> Self {
        self.state.stencil_back = state;
        self
    }

    /// Set both front and back stencil states.
    pub fn stencil_both(mut self, state: StencilFaceStateDescriptor) -> Self {
        self.state.stencil_front = state;
        self.state.stencil_back = state;
        self
    }

    /// Set the stencil read mask.
    pub fn stencil_read_mask(mut self, mask: u32) -> Self {
        self.state.stencil_read_mask = mask;
        self
    }

    /// Set the stencil write mask.
    pub fn stencil_write_mask(mut self, mask: u32) -> Self {
        self.state.stencil_write_mask = mask;
        self
    }

    /// Set both stencil masks.
    pub fn stencil_masks(mut self, read_mask: u32, write_mask: u32) -> Self {
        self.state.stencil_read_mask = read_mask;
        self.state.stencil_write_mask = write_mask;
        self
    }

    /// Set the depth bias configuration.
    pub fn bias(mut self, bias: DepthBiasStateDescriptor) -> Self {
        self.state.bias = bias;
        self
    }

    /// Set depth bias constant.
    pub fn bias_constant(mut self, constant: i32) -> Self {
        self.state.bias.constant = constant;
        self
    }

    /// Set depth bias slope scale.
    pub fn bias_slope_scale(mut self, slope_scale: f32) -> Self {
        self.state.bias.slope_scale = slope_scale;
        self
    }

    /// Set depth bias clamp.
    pub fn bias_clamp(mut self, clamp: f32) -> Self {
        self.state.bias.clamp = clamp;
        self
    }

    /// Build the depth-stencil state descriptor.
    pub fn build(self) -> DepthStencilStateDescriptor {
        self.state
    }

    /// Build and convert to wgpu type.
    pub fn build_wgpu(self) -> wgpu::DepthStencilState {
        self.state.into()
    }
}

// ---------------------------------------------------------------------------
// Info Structs
// ---------------------------------------------------------------------------

/// Information about a depth comparison function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompareFunctionInfo {
    /// The wgpu compare function.
    pub function: wgpu::CompareFunction,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of when the test passes.
    pub description: &'static str,
    /// Typical use cases.
    pub use_cases: &'static [&'static str],
}

/// All 8 comparison functions with documentation.
pub const COMPARE_FUNCTIONS: [CompareFunctionInfo; 8] = [
    CompareFunctionInfo {
        function: wgpu::CompareFunction::Never,
        name: "Never",
        description: "Test never passes",
        use_cases: &["discard all fragments", "stencil-only passes"],
    },
    CompareFunctionInfo {
        function: wgpu::CompareFunction::Less,
        name: "Less",
        description: "Passes when fragment < buffer",
        use_cases: &["standard forward rendering", "default depth test"],
    },
    CompareFunctionInfo {
        function: wgpu::CompareFunction::Equal,
        name: "Equal",
        description: "Passes when fragment == buffer",
        use_cases: &["decals", "exact depth matching", "stencil testing"],
    },
    CompareFunctionInfo {
        function: wgpu::CompareFunction::LessEqual,
        name: "LessEqual",
        description: "Passes when fragment <= buffer",
        use_cases: &["common default", "coplanar geometry"],
    },
    CompareFunctionInfo {
        function: wgpu::CompareFunction::Greater,
        name: "Greater",
        description: "Passes when fragment > buffer",
        use_cases: &["reverse-Z rendering", "back-to-front sorting"],
    },
    CompareFunctionInfo {
        function: wgpu::CompareFunction::NotEqual,
        name: "NotEqual",
        description: "Passes when fragment != buffer",
        use_cases: &["outline effects", "stencil masking"],
    },
    CompareFunctionInfo {
        function: wgpu::CompareFunction::GreaterEqual,
        name: "GreaterEqual",
        description: "Passes when fragment >= buffer",
        use_cases: &["reverse-Z with equal", "coplanar reverse-Z"],
    },
    CompareFunctionInfo {
        function: wgpu::CompareFunction::Always,
        name: "Always",
        description: "Test always passes",
        use_cases: &["disable depth test", "full-screen effects", "UI rendering"],
    },
];

/// Get compare function info by function type.
pub fn get_compare_function_info(function: wgpu::CompareFunction) -> Option<&'static CompareFunctionInfo> {
    COMPARE_FUNCTIONS.iter().find(|info| info.function == function)
}

/// Information about a depth texture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthFormatInfo {
    /// The wgpu texture format.
    pub format: wgpu::TextureFormat,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of the format.
    pub description: &'static str,
    /// Whether this format has a stencil component.
    pub has_stencil: bool,
    /// Depth bit depth (approximate).
    pub depth_bits: u32,
    /// Stencil bit depth (0 if no stencil).
    pub stencil_bits: u32,
}

/// All 4 depth formats with documentation.
pub const DEPTH_FORMATS: [DepthFormatInfo; 4] = [
    DepthFormatInfo {
        format: wgpu::TextureFormat::Depth32Float,
        name: "Depth32Float",
        description: "32-bit floating point depth, no stencil",
        has_stencil: false,
        depth_bits: 32,
        stencil_bits: 0,
    },
    DepthFormatInfo {
        format: wgpu::TextureFormat::Depth24Plus,
        name: "Depth24Plus",
        description: "At least 24-bit depth, no stencil",
        has_stencil: false,
        depth_bits: 24,
        stencil_bits: 0,
    },
    DepthFormatInfo {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        name: "Depth24PlusStencil8",
        description: "At least 24-bit depth + 8-bit stencil",
        has_stencil: true,
        depth_bits: 24,
        stencil_bits: 8,
    },
    DepthFormatInfo {
        format: wgpu::TextureFormat::Depth32FloatStencil8,
        name: "Depth32FloatStencil8",
        description: "32-bit floating point depth + 8-bit stencil",
        has_stencil: true,
        depth_bits: 32,
        stencil_bits: 8,
    },
];

/// Get depth format info by format type.
pub fn get_depth_format_info(format: wgpu::TextureFormat) -> Option<&'static DepthFormatInfo> {
    DEPTH_FORMATS.iter().find(|info| info.format == format)
}

/// Information about a depth preset configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthPresetInfo {
    /// Human-readable name.
    pub name: &'static str,
    /// Description of the preset.
    pub description: &'static str,
    /// The depth comparison function used.
    pub depth_compare: wgpu::CompareFunction,
    /// Whether depth writing is enabled.
    pub depth_write_enabled: bool,
    /// Typical use cases.
    pub use_cases: &'static [&'static str],
}

/// All 11 depth presets with documentation.
pub const DEPTH_PRESETS: [DepthPresetInfo; 11] = [
    DepthPresetInfo {
        name: "depth_less",
        description: "Standard forward rendering with Less comparison",
        depth_compare: wgpu::CompareFunction::Less,
        depth_write_enabled: true,
        use_cases: &["opaque geometry", "standard rendering"],
    },
    DepthPresetInfo {
        name: "depth_less_equal",
        description: "Less-equal comparison for coplanar geometry",
        depth_compare: wgpu::CompareFunction::LessEqual,
        depth_write_enabled: true,
        use_cases: &["coplanar geometry", "common default"],
    },
    DepthPresetInfo {
        name: "depth_always",
        description: "Depth test disabled",
        depth_compare: wgpu::CompareFunction::Always,
        depth_write_enabled: false,
        use_cases: &["full-screen effects", "UI", "post-processing"],
    },
    DepthPresetInfo {
        name: "depth_greater",
        description: "Reverse-Z rendering",
        depth_compare: wgpu::CompareFunction::Greater,
        depth_write_enabled: true,
        use_cases: &["reverse-Z", "better depth precision"],
    },
    DepthPresetInfo {
        name: "depth_greater_equal",
        description: "Reverse-Z with equal comparison",
        depth_compare: wgpu::CompareFunction::GreaterEqual,
        depth_write_enabled: true,
        use_cases: &["reverse-Z coplanar", "decals with reverse-Z"],
    },
    DepthPresetInfo {
        name: "depth_never",
        description: "All fragments fail depth test",
        depth_compare: wgpu::CompareFunction::Never,
        depth_write_enabled: false,
        use_cases: &["stencil-only passes", "debug"],
    },
    DepthPresetInfo {
        name: "depth_read_only",
        description: "Test but don't write depth",
        depth_compare: wgpu::CompareFunction::Less,
        depth_write_enabled: false,
        use_cases: &["transparent objects", "particles"],
    },
    DepthPresetInfo {
        name: "depth_read_only_reverse_z",
        description: "Read-only reverse-Z",
        depth_compare: wgpu::CompareFunction::Greater,
        depth_write_enabled: false,
        use_cases: &["transparent with reverse-Z"],
    },
    DepthPresetInfo {
        name: "transparent",
        description: "Transparent object rendering",
        depth_compare: wgpu::CompareFunction::LessEqual,
        depth_write_enabled: false,
        use_cases: &["alpha blending", "glass", "water"],
    },
    DepthPresetInfo {
        name: "shadow_map",
        description: "Shadow map generation",
        depth_compare: wgpu::CompareFunction::Less,
        depth_write_enabled: true,
        use_cases: &["shadow pass", "depth-only rendering"],
    },
    DepthPresetInfo {
        name: "depth_prepass",
        description: "Early-Z depth pre-pass",
        depth_compare: wgpu::CompareFunction::Less,
        depth_write_enabled: true,
        use_cases: &["Z-prepass", "occlusion culling prep"],
    },
];

/// Get depth preset info by name.
pub fn get_depth_preset_info(name: &str) -> Option<&'static DepthPresetInfo> {
    DEPTH_PRESETS.iter().find(|info| info.name == name)
}

/// Information about a stencil operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilOperationInfo {
    /// The wgpu stencil operation.
    pub operation: wgpu::StencilOperation,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of the operation.
    pub description: &'static str,
    /// Typical use cases.
    pub use_cases: &'static [&'static str],
}

/// All 8 stencil operations with documentation.
pub const STENCIL_OPERATIONS: [StencilOperationInfo; 8] = [
    StencilOperationInfo {
        operation: wgpu::StencilOperation::Keep,
        name: "Keep",
        description: "Keep the current stencil value",
        use_cases: &["no change", "read-only stencil test"],
    },
    StencilOperationInfo {
        operation: wgpu::StencilOperation::Zero,
        name: "Zero",
        description: "Set stencil value to 0",
        use_cases: &["clear stencil", "reset regions"],
    },
    StencilOperationInfo {
        operation: wgpu::StencilOperation::Replace,
        name: "Replace",
        description: "Replace with reference value",
        use_cases: &["write mask", "mark regions"],
    },
    StencilOperationInfo {
        operation: wgpu::StencilOperation::IncrementClamp,
        name: "IncrementClamp",
        description: "Increment, clamp at maximum",
        use_cases: &["shadow volumes", "counting overlaps"],
    },
    StencilOperationInfo {
        operation: wgpu::StencilOperation::DecrementClamp,
        name: "DecrementClamp",
        description: "Decrement, clamp at 0",
        use_cases: &["shadow volumes exit", "undo increment"],
    },
    StencilOperationInfo {
        operation: wgpu::StencilOperation::Invert,
        name: "Invert",
        description: "Bitwise invert stencil value",
        use_cases: &["XOR masking", "toggle regions"],
    },
    StencilOperationInfo {
        operation: wgpu::StencilOperation::IncrementWrap,
        name: "IncrementWrap",
        description: "Increment, wrap to 0 at overflow",
        use_cases: &["counting with wrap", "modular arithmetic"],
    },
    StencilOperationInfo {
        operation: wgpu::StencilOperation::DecrementWrap,
        name: "DecrementWrap",
        description: "Decrement, wrap to max at underflow",
        use_cases: &["counting with wrap", "modular arithmetic"],
    },
];

/// Get stencil operation info by operation type.
pub fn get_stencil_operation_info(operation: wgpu::StencilOperation) -> Option<&'static StencilOperationInfo> {
    STENCIL_OPERATIONS.iter().find(|info| info.operation == operation)
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Check if a texture format is a depth format.
pub fn is_depth_format(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Depth32Float
            | wgpu::TextureFormat::Depth24Plus
            | wgpu::TextureFormat::Depth24PlusStencil8
            | wgpu::TextureFormat::Depth32FloatStencil8
    )
}

/// Check if a texture format has a stencil component.
pub fn has_stencil(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Depth24PlusStencil8 | wgpu::TextureFormat::Depth32FloatStencil8
    )
}

/// Get all depth format names.
pub fn depth_format_names() -> impl Iterator<Item = &'static str> {
    DEPTH_FORMATS.iter().map(|info| info.name)
}

/// Get all depth preset names.
pub fn depth_preset_names() -> impl Iterator<Item = &'static str> {
    DEPTH_PRESETS.iter().map(|info| info.name)
}

/// Get all compare function names.
pub fn compare_function_names() -> impl Iterator<Item = &'static str> {
    COMPARE_FUNCTIONS.iter().map(|info| info.name)
}

/// Get all stencil operation names.
pub fn stencil_operation_names() -> impl Iterator<Item = &'static str> {
    STENCIL_OPERATIONS.iter().map(|info| info.name)
}

/// Get depth presets that support stencil operations.
pub fn stencil_capable_presets() -> impl Iterator<Item = &'static DepthPresetInfo> {
    // All presets are stencil-capable if used with stencil format
    DEPTH_PRESETS.iter()
}

// ---------------------------------------------------------------------------
// Display Implementations
// ---------------------------------------------------------------------------

impl fmt::Display for DepthStencilStateDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DepthStencilState {{ format: {:?}, depth_write: {}, depth_compare: {:?} }}",
            self.format, self.depth_write_enabled, self.depth_compare
        )
    }
}

impl fmt::Display for StencilFaceStateDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StencilFace {{ compare: {:?}, fail: {:?}, depth_fail: {:?}, pass: {:?} }}",
            self.compare, self.fail_op, self.depth_fail_op, self.pass_op
        )
    }
}

impl fmt::Display for DepthBiasStateDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_active() {
            write!(
                f,
                "DepthBias {{ constant: {}, slope: {}, clamp: {} }}",
                self.constant, self.slope_scale, self.clamp
            )
        } else {
            write!(f, "DepthBias(none)")
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DepthStencilStateDescriptor Basic Tests
    // =========================================================================

    #[test]
    fn test_default() {
        let state = DepthStencilStateDescriptor::default();
        assert_eq!(state.format, wgpu::TextureFormat::Depth24PlusStencil8);
        assert!(state.depth_write_enabled);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
        assert_eq!(state.stencil_read_mask, 0xFFFFFFFF);
        assert_eq!(state.stencil_write_mask, 0xFFFFFFFF);
    }

    #[test]
    fn test_new() {
        let state = DepthStencilStateDescriptor::new();
        assert_eq!(state, DepthStencilStateDescriptor::default());
    }

    // =========================================================================
    // Depth Preset Tests (T-WGPU-P3.6.1)
    // =========================================================================

    #[test]
    fn test_depth_less() {
        let state = DepthStencilStateDescriptor::depth_less();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_depth_less_equal() {
        let state = DepthStencilStateDescriptor::depth_less_equal();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_depth_always() {
        let state = DepthStencilStateDescriptor::depth_always();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Always);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_depth_greater() {
        let state = DepthStencilStateDescriptor::depth_greater();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_depth_greater_equal() {
        let state = DepthStencilStateDescriptor::depth_greater_equal();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::GreaterEqual);
        assert!(state.depth_write_enabled);
    }

    #[test]
    fn test_depth_never() {
        let state = DepthStencilStateDescriptor::depth_never();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Never);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_depth_read_only() {
        let state = DepthStencilStateDescriptor::depth_read_only();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_depth_read_only_reverse_z() {
        let state = DepthStencilStateDescriptor::depth_read_only_reverse_z();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_transparent() {
        let state = DepthStencilStateDescriptor::transparent();
        assert_eq!(state.depth_compare, wgpu::CompareFunction::LessEqual);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_shadow_map() {
        let state = DepthStencilStateDescriptor::shadow_map();
        assert_eq!(state.format, wgpu::TextureFormat::Depth32Float);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
        assert!(state.depth_write_enabled);
        assert!(state.bias.is_active());
    }

    #[test]
    fn test_depth_prepass() {
        let state = DepthStencilStateDescriptor::depth_prepass();
        assert_eq!(state.format, wgpu::TextureFormat::Depth32Float);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Less);
        assert!(state.depth_write_enabled);
    }

    // =========================================================================
    // Stencil Preset Tests (T-WGPU-P3.6.2)
    // =========================================================================

    #[test]
    fn test_stencil_write() {
        let state = DepthStencilStateDescriptor::stencil_write();
        assert_eq!(state.stencil_front.pass_op, wgpu::StencilOperation::Replace);
        assert_eq!(state.stencil_back.pass_op, wgpu::StencilOperation::Replace);
        assert_eq!(state.stencil_front.compare, wgpu::CompareFunction::Always);
    }

    #[test]
    fn test_stencil_replace() {
        let state = DepthStencilStateDescriptor::stencil_replace();
        assert_eq!(state.stencil_front.fail_op, wgpu::StencilOperation::Replace);
        assert_eq!(state.stencil_front.depth_fail_op, wgpu::StencilOperation::Replace);
        assert_eq!(state.stencil_front.pass_op, wgpu::StencilOperation::Replace);
    }

    #[test]
    fn test_stencil_increment() {
        let state = DepthStencilStateDescriptor::stencil_increment();
        assert_eq!(state.stencil_front.pass_op, wgpu::StencilOperation::IncrementClamp);
        assert_eq!(state.stencil_back.pass_op, wgpu::StencilOperation::IncrementClamp);
    }

    #[test]
    fn test_stencil_decrement() {
        let state = DepthStencilStateDescriptor::stencil_decrement();
        assert_eq!(state.stencil_front.pass_op, wgpu::StencilOperation::DecrementClamp);
        assert_eq!(state.stencil_back.pass_op, wgpu::StencilOperation::DecrementClamp);
    }

    #[test]
    fn test_stencil_invert() {
        let state = DepthStencilStateDescriptor::stencil_invert();
        assert_eq!(state.stencil_front.pass_op, wgpu::StencilOperation::Invert);
        assert_eq!(state.stencil_back.pass_op, wgpu::StencilOperation::Invert);
    }

    #[test]
    fn test_stencil_zero() {
        let state = DepthStencilStateDescriptor::stencil_zero();
        assert_eq!(state.stencil_front.pass_op, wgpu::StencilOperation::Zero);
        assert_eq!(state.stencil_back.pass_op, wgpu::StencilOperation::Zero);
    }

    #[test]
    fn test_stencil_read_equal() {
        let state = DepthStencilStateDescriptor::stencil_read_equal();
        assert_eq!(state.stencil_front.compare, wgpu::CompareFunction::Equal);
        assert_eq!(state.stencil_front.pass_op, wgpu::StencilOperation::Keep);
    }

    #[test]
    fn test_stencil_read_not_equal() {
        let state = DepthStencilStateDescriptor::stencil_read_not_equal();
        assert_eq!(state.stencil_front.compare, wgpu::CompareFunction::NotEqual);
        assert_eq!(state.stencil_front.pass_op, wgpu::StencilOperation::Keep);
    }

    // =========================================================================
    // Fluent API Tests
    // =========================================================================

    #[test]
    fn test_fluent_format() {
        let state = DepthStencilStateDescriptor::new()
            .format(wgpu::TextureFormat::Depth32Float);
        assert_eq!(state.format, wgpu::TextureFormat::Depth32Float);
    }

    #[test]
    fn test_fluent_depth_write() {
        let state = DepthStencilStateDescriptor::new()
            .depth_write_enabled(false);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_fluent_depth_compare() {
        let state = DepthStencilStateDescriptor::new()
            .depth_compare(wgpu::CompareFunction::Greater);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
    }

    #[test]
    fn test_fluent_stencil_masks() {
        let state = DepthStencilStateDescriptor::new()
            .stencil_read_mask(0xFF)
            .stencil_write_mask(0x0F);
        assert_eq!(state.stencil_read_mask, 0xFF);
        assert_eq!(state.stencil_write_mask, 0x0F);
    }

    #[test]
    fn test_fluent_stencil_both() {
        let stencil = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Equal);
        let state = DepthStencilStateDescriptor::new()
            .stencil_both(stencil);
        assert_eq!(state.stencil_front.compare, wgpu::CompareFunction::Equal);
        assert_eq!(state.stencil_back.compare, wgpu::CompareFunction::Equal);
    }

    #[test]
    fn test_fluent_bias() {
        let bias = DepthBiasStateDescriptor::shadow_map();
        let state = DepthStencilStateDescriptor::new().bias(bias);
        assert!(state.bias.is_active());
    }

    // =========================================================================
    // Query Method Tests
    // =========================================================================

    #[test]
    fn test_is_depth_test_enabled() {
        assert!(DepthStencilStateDescriptor::depth_less().is_depth_test_enabled());
        assert!(!DepthStencilStateDescriptor::depth_always().is_depth_test_enabled());

        // Depth write enabled but always compare
        let state = DepthStencilStateDescriptor::new()
            .depth_compare(wgpu::CompareFunction::Always)
            .depth_write_enabled(true);
        assert!(state.is_depth_test_enabled());
    }

    #[test]
    fn test_is_stencil_test_enabled() {
        assert!(!DepthStencilStateDescriptor::depth_less().is_stencil_test_enabled());
        assert!(DepthStencilStateDescriptor::stencil_write().is_stencil_test_enabled());
    }

    #[test]
    fn test_has_stencil_format() {
        assert!(DepthStencilStateDescriptor::depth_less().has_stencil_format());
        assert!(!DepthStencilStateDescriptor::shadow_map().has_stencil_format());
    }

    #[test]
    fn test_is_depth_format_method() {
        assert!(DepthStencilStateDescriptor::depth_less().is_depth_format());
        assert!(DepthStencilStateDescriptor::shadow_map().is_depth_format());
    }

    #[test]
    fn test_has_depth_bias() {
        assert!(!DepthStencilStateDescriptor::depth_less().has_depth_bias());
        assert!(DepthStencilStateDescriptor::shadow_map().has_depth_bias());
    }

    // =========================================================================
    // StencilFaceStateDescriptor Tests
    // =========================================================================

    #[test]
    fn test_stencil_face_default() {
        let face = StencilFaceStateDescriptor::default();
        assert_eq!(face.compare, wgpu::CompareFunction::Always);
        assert_eq!(face.fail_op, wgpu::StencilOperation::Keep);
        assert_eq!(face.depth_fail_op, wgpu::StencilOperation::Keep);
        assert_eq!(face.pass_op, wgpu::StencilOperation::Keep);
    }

    #[test]
    fn test_stencil_face_disabled() {
        let face = StencilFaceStateDescriptor::disabled();
        assert!(face.is_disabled());
    }

    #[test]
    fn test_stencil_face_is_disabled() {
        let disabled = StencilFaceStateDescriptor::default();
        assert!(disabled.is_disabled());

        let active = StencilFaceStateDescriptor::new()
            .pass_op(wgpu::StencilOperation::Replace);
        assert!(!active.is_disabled());
    }

    #[test]
    fn test_stencil_face_modifies_stencil() {
        let keeps = StencilFaceStateDescriptor::default();
        assert!(!keeps.modifies_stencil());

        let modifies = StencilFaceStateDescriptor::new()
            .pass_op(wgpu::StencilOperation::Replace);
        assert!(modifies.modifies_stencil());
    }

    #[test]
    fn test_stencil_face_all_ops() {
        let face = StencilFaceStateDescriptor::new()
            .all_ops(wgpu::StencilOperation::Replace);
        assert_eq!(face.fail_op, wgpu::StencilOperation::Replace);
        assert_eq!(face.depth_fail_op, wgpu::StencilOperation::Replace);
        assert_eq!(face.pass_op, wgpu::StencilOperation::Replace);
    }

    #[test]
    fn test_stencil_face_fluent() {
        let face = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Equal)
            .fail_op(wgpu::StencilOperation::Zero)
            .depth_fail_op(wgpu::StencilOperation::Keep)
            .pass_op(wgpu::StencilOperation::Replace);

        assert_eq!(face.compare, wgpu::CompareFunction::Equal);
        assert_eq!(face.fail_op, wgpu::StencilOperation::Zero);
        assert_eq!(face.depth_fail_op, wgpu::StencilOperation::Keep);
        assert_eq!(face.pass_op, wgpu::StencilOperation::Replace);
    }

    // =========================================================================
    // DepthBiasStateDescriptor Tests
    // =========================================================================

    #[test]
    fn test_depth_bias_default() {
        let bias = DepthBiasStateDescriptor::default();
        assert_eq!(bias.constant, 0);
        assert_eq!(bias.slope_scale, 0.0);
        assert_eq!(bias.clamp, 0.0);
        assert!(!bias.is_active());
    }

    #[test]
    fn test_depth_bias_none() {
        let bias = DepthBiasStateDescriptor::none();
        assert!(!bias.is_active());
    }

    #[test]
    fn test_depth_bias_shadow_map() {
        let bias = DepthBiasStateDescriptor::shadow_map();
        assert_eq!(bias.constant, 2);
        assert_eq!(bias.slope_scale, 2.0);
        assert!(bias.is_active());
    }

    #[test]
    fn test_depth_bias_cascaded() {
        let bias = DepthBiasStateDescriptor::cascaded_shadow_map();
        assert_eq!(bias.constant, 4);
        assert_eq!(bias.slope_scale, 3.0);
    }

    #[test]
    fn test_depth_bias_decal() {
        let bias = DepthBiasStateDescriptor::decal();
        assert_eq!(bias.constant, 1);
        assert_eq!(bias.slope_scale, 1.0);
    }

    #[test]
    fn test_depth_bias_outline() {
        let bias = DepthBiasStateDescriptor::outline();
        assert_eq!(bias.constant, -1);
        assert_eq!(bias.slope_scale, -1.0);
    }

    #[test]
    fn test_depth_bias_fluent() {
        let bias = DepthBiasStateDescriptor::new()
            .constant(5)
            .slope_scale(2.5)
            .clamp(0.01);

        assert_eq!(bias.constant, 5);
        assert_eq!(bias.slope_scale, 2.5);
        assert_eq!(bias.clamp, 0.01);
    }

    // =========================================================================
    // Builder Tests
    // =========================================================================

    #[test]
    fn test_builder_default() {
        let state = DepthStencilStateBuilder::new().build();
        assert_eq!(state, DepthStencilStateDescriptor::default());
    }

    #[test]
    fn test_builder_from_preset() {
        let state = DepthStencilStateBuilder::from_preset(DepthStencilStateDescriptor::depth_greater())
            .depth_write_enabled(false)
            .build();

        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
        assert!(!state.depth_write_enabled);
    }

    #[test]
    fn test_builder_stencil_masks() {
        let state = DepthStencilStateBuilder::new()
            .stencil_masks(0xFF, 0x0F)
            .build();

        assert_eq!(state.stencil_read_mask, 0xFF);
        assert_eq!(state.stencil_write_mask, 0x0F);
    }

    #[test]
    fn test_builder_bias_components() {
        let state = DepthStencilStateBuilder::new()
            .bias_constant(3)
            .bias_slope_scale(1.5)
            .bias_clamp(0.02)
            .build();

        assert_eq!(state.bias.constant, 3);
        assert_eq!(state.bias.slope_scale, 1.5);
        assert_eq!(state.bias.clamp, 0.02);
    }

    #[test]
    fn test_builder_build_wgpu() {
        let wgpu_state = DepthStencilStateBuilder::new()
            .format(wgpu::TextureFormat::Depth32Float)
            .depth_compare(wgpu::CompareFunction::Greater)
            .build_wgpu();

        assert_eq!(wgpu_state.format, wgpu::TextureFormat::Depth32Float);
        assert_eq!(wgpu_state.depth_compare, wgpu::CompareFunction::Greater);
    }

    // =========================================================================
    // Conversion Tests
    // =========================================================================

    #[test]
    fn test_into_wgpu_depth_stencil_state() {
        let desc = DepthStencilStateDescriptor::depth_less();
        let wgpu_state: wgpu::DepthStencilState = desc.into();

        assert_eq!(wgpu_state.format, desc.format);
        assert_eq!(wgpu_state.depth_write_enabled, desc.depth_write_enabled);
        assert_eq!(wgpu_state.depth_compare, desc.depth_compare);
    }

    #[test]
    fn test_from_wgpu_depth_stencil_state() {
        let wgpu_state = wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Greater,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let desc: DepthStencilStateDescriptor = wgpu_state.into();
        assert_eq!(desc.format, wgpu::TextureFormat::Depth32Float);
        assert!(!desc.depth_write_enabled);
        assert_eq!(desc.depth_compare, wgpu::CompareFunction::Greater);
    }

    #[test]
    fn test_stencil_face_into_wgpu() {
        let face = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Equal)
            .pass_op(wgpu::StencilOperation::Replace);

        let wgpu_face: wgpu::StencilFaceState = face.into();
        assert_eq!(wgpu_face.compare, wgpu::CompareFunction::Equal);
        assert_eq!(wgpu_face.pass_op, wgpu::StencilOperation::Replace);
    }

    #[test]
    fn test_depth_bias_into_wgpu() {
        let bias = DepthBiasStateDescriptor::shadow_map();
        let wgpu_bias: wgpu::DepthBiasState = bias.into();

        assert_eq!(wgpu_bias.constant, 2);
        assert_eq!(wgpu_bias.slope_scale, 2.0);
    }

    // =========================================================================
    // Info Struct Tests
    // =========================================================================

    #[test]
    fn test_compare_functions_count() {
        assert_eq!(COMPARE_FUNCTIONS.len(), 8);
    }

    #[test]
    fn test_get_compare_function_info() {
        let info = get_compare_function_info(wgpu::CompareFunction::Less);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Less");
        assert!(info.description.contains("fragment < buffer"));
    }

    #[test]
    fn test_all_compare_functions_have_info() {
        let functions = [
            wgpu::CompareFunction::Never,
            wgpu::CompareFunction::Less,
            wgpu::CompareFunction::Equal,
            wgpu::CompareFunction::LessEqual,
            wgpu::CompareFunction::Greater,
            wgpu::CompareFunction::NotEqual,
            wgpu::CompareFunction::GreaterEqual,
            wgpu::CompareFunction::Always,
        ];
        for func in functions {
            assert!(
                get_compare_function_info(func).is_some(),
                "Missing info for {:?}",
                func
            );
        }
    }

    #[test]
    fn test_depth_formats_count() {
        assert_eq!(DEPTH_FORMATS.len(), 4);
    }

    #[test]
    fn test_get_depth_format_info() {
        let info = get_depth_format_info(wgpu::TextureFormat::Depth24PlusStencil8);
        assert!(info.is_some());
        let info = info.unwrap();
        assert!(info.has_stencil);
        assert_eq!(info.stencil_bits, 8);
    }

    #[test]
    fn test_depth_presets_count() {
        assert_eq!(DEPTH_PRESETS.len(), 11);
    }

    #[test]
    fn test_get_depth_preset_info() {
        let info = get_depth_preset_info("depth_less");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.depth_compare, wgpu::CompareFunction::Less);
        assert!(info.depth_write_enabled);
    }

    #[test]
    fn test_stencil_operations_count() {
        assert_eq!(STENCIL_OPERATIONS.len(), 8);
    }

    #[test]
    fn test_get_stencil_operation_info() {
        let info = get_stencil_operation_info(wgpu::StencilOperation::Replace);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Replace");
    }

    #[test]
    fn test_all_stencil_operations_have_info() {
        let operations = [
            wgpu::StencilOperation::Keep,
            wgpu::StencilOperation::Zero,
            wgpu::StencilOperation::Replace,
            wgpu::StencilOperation::IncrementClamp,
            wgpu::StencilOperation::DecrementClamp,
            wgpu::StencilOperation::Invert,
            wgpu::StencilOperation::IncrementWrap,
            wgpu::StencilOperation::DecrementWrap,
        ];
        for op in operations {
            assert!(
                get_stencil_operation_info(op).is_some(),
                "Missing info for {:?}",
                op
            );
        }
    }

    // =========================================================================
    // Helper Function Tests
    // =========================================================================

    #[test]
    fn test_is_depth_format() {
        assert!(is_depth_format(wgpu::TextureFormat::Depth32Float));
        assert!(is_depth_format(wgpu::TextureFormat::Depth24Plus));
        assert!(is_depth_format(wgpu::TextureFormat::Depth24PlusStencil8));
        assert!(is_depth_format(wgpu::TextureFormat::Depth32FloatStencil8));
        assert!(!is_depth_format(wgpu::TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn test_has_stencil() {
        assert!(!has_stencil(wgpu::TextureFormat::Depth32Float));
        assert!(!has_stencil(wgpu::TextureFormat::Depth24Plus));
        assert!(has_stencil(wgpu::TextureFormat::Depth24PlusStencil8));
        assert!(has_stencil(wgpu::TextureFormat::Depth32FloatStencil8));
    }

    #[test]
    fn test_depth_format_names() {
        let names: Vec<_> = depth_format_names().collect();
        assert_eq!(names.len(), 4);
        assert!(names.contains(&"Depth32Float"));
        assert!(names.contains(&"Depth24PlusStencil8"));
    }

    #[test]
    fn test_depth_preset_names() {
        let names: Vec<_> = depth_preset_names().collect();
        assert_eq!(names.len(), 11);
        assert!(names.contains(&"depth_less"));
        assert!(names.contains(&"shadow_map"));
    }

    #[test]
    fn test_compare_function_names() {
        let names: Vec<_> = compare_function_names().collect();
        assert_eq!(names.len(), 8);
        assert!(names.contains(&"Less"));
        assert!(names.contains(&"Always"));
    }

    #[test]
    fn test_stencil_operation_names() {
        let names: Vec<_> = stencil_operation_names().collect();
        assert_eq!(names.len(), 8);
        assert!(names.contains(&"Keep"));
        assert!(names.contains(&"Replace"));
    }

    // =========================================================================
    // Thread Safety Tests
    // =========================================================================

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<DepthStencilStateDescriptor>();
        assert_sync::<DepthStencilStateDescriptor>();
        assert_send::<StencilFaceStateDescriptor>();
        assert_sync::<StencilFaceStateDescriptor>();
        assert_send::<DepthBiasStateDescriptor>();
        assert_sync::<DepthBiasStateDescriptor>();
    }

    // =========================================================================
    // Display Tests
    // =========================================================================

    #[test]
    fn test_depth_stencil_display() {
        let state = DepthStencilStateDescriptor::depth_less();
        let display = format!("{}", state);
        assert!(display.contains("DepthStencilState"));
        assert!(display.contains("Less"));
    }

    #[test]
    fn test_stencil_face_display() {
        let face = StencilFaceStateDescriptor::new()
            .pass_op(wgpu::StencilOperation::Replace);
        let display = format!("{}", face);
        assert!(display.contains("StencilFace"));
        assert!(display.contains("Replace"));
    }

    #[test]
    fn test_depth_bias_display_active() {
        let bias = DepthBiasStateDescriptor::shadow_map();
        let display = format!("{}", bias);
        assert!(display.contains("DepthBias"));
        assert!(display.contains("constant"));
    }

    #[test]
    fn test_depth_bias_display_none() {
        let bias = DepthBiasStateDescriptor::none();
        let display = format!("{}", bias);
        assert!(display.contains("none"));
    }

    // =========================================================================
    // Clone and Equality Tests
    // =========================================================================

    #[test]
    fn test_clone() {
        let state = DepthStencilStateDescriptor::depth_less();
        let cloned = state;
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_equality() {
        let a = DepthStencilStateDescriptor::depth_less();
        let b = DepthStencilStateDescriptor::depth_less();
        let c = DepthStencilStateDescriptor::depth_greater();

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // =========================================================================
    // Additional Whitebox Tests - Full Coverage
    // =========================================================================

    #[test]
    fn test_all_depth_presets_exist() {
        let presets = [
            DepthStencilStateDescriptor::depth_less(),
            DepthStencilStateDescriptor::depth_less_equal(),
            DepthStencilStateDescriptor::depth_always(),
            DepthStencilStateDescriptor::depth_greater(),
            DepthStencilStateDescriptor::depth_greater_equal(),
            DepthStencilStateDescriptor::depth_never(),
            DepthStencilStateDescriptor::depth_read_only(),
            DepthStencilStateDescriptor::depth_read_only_reverse_z(),
            DepthStencilStateDescriptor::transparent(),
            DepthStencilStateDescriptor::shadow_map(),
            DepthStencilStateDescriptor::depth_prepass(),
        ];
        assert_eq!(presets.len(), 11);
    }

    #[test]
    fn test_all_stencil_presets_exist() {
        let presets = [
            DepthStencilStateDescriptor::stencil_write(),
            DepthStencilStateDescriptor::stencil_replace(),
            DepthStencilStateDescriptor::stencil_increment(),
            DepthStencilStateDescriptor::stencil_decrement(),
            DepthStencilStateDescriptor::stencil_invert(),
            DepthStencilStateDescriptor::stencil_zero(),
            DepthStencilStateDescriptor::stencil_read_equal(),
            DepthStencilStateDescriptor::stencil_read_not_equal(),
        ];
        assert_eq!(presets.len(), 8);
    }

    #[test]
    fn test_preset_distinctness() {
        let presets = [
            DepthStencilStateDescriptor::depth_less(),
            DepthStencilStateDescriptor::depth_greater(),
            DepthStencilStateDescriptor::depth_always(),
            DepthStencilStateDescriptor::shadow_map(),
        ];

        for i in 0..presets.len() {
            for j in (i + 1)..presets.len() {
                assert_ne!(presets[i], presets[j], "Presets {} and {} should differ", i, j);
            }
        }
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original = DepthStencilStateDescriptor::depth_less();
        let wgpu_state: wgpu::DepthStencilState = original.into();
        let back: DepthStencilStateDescriptor = wgpu_state.into();
        assert_eq!(original, back);
    }

    #[test]
    fn test_stencil_face_roundtrip() {
        let original = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Equal)
            .fail_op(wgpu::StencilOperation::Zero)
            .pass_op(wgpu::StencilOperation::Replace);

        let wgpu_face: wgpu::StencilFaceState = original.into();
        let back: StencilFaceStateDescriptor = wgpu_face.into();
        assert_eq!(original, back);
    }

    #[test]
    fn test_depth_bias_roundtrip() {
        let original = DepthBiasStateDescriptor::shadow_map();
        let wgpu_bias: wgpu::DepthBiasState = original.into();
        let back: DepthBiasStateDescriptor = wgpu_bias.into();
        assert_eq!(original, back);
    }

    #[test]
    fn test_builder_chain_all_options() {
        let stencil = StencilFaceStateDescriptor::new()
            .compare(wgpu::CompareFunction::Equal)
            .pass_op(wgpu::StencilOperation::Replace);

        let state = DepthStencilStateBuilder::new()
            .format(wgpu::TextureFormat::Depth32FloatStencil8)
            .depth_write_enabled(false)
            .depth_compare(wgpu::CompareFunction::Greater)
            .stencil_front(stencil)
            .stencil_back(stencil)
            .stencil_read_mask(0xFF)
            .stencil_write_mask(0x0F)
            .bias_constant(4)
            .bias_slope_scale(2.0)
            .bias_clamp(0.01)
            .build();

        assert_eq!(state.format, wgpu::TextureFormat::Depth32FloatStencil8);
        assert!(!state.depth_write_enabled);
        assert_eq!(state.depth_compare, wgpu::CompareFunction::Greater);
        assert_eq!(state.stencil_front.compare, wgpu::CompareFunction::Equal);
        assert_eq!(state.stencil_read_mask, 0xFF);
        assert_eq!(state.stencil_write_mask, 0x0F);
        assert_eq!(state.bias.constant, 4);
    }

    #[test]
    fn test_info_use_cases_non_empty() {
        for info in &COMPARE_FUNCTIONS {
            assert!(!info.use_cases.is_empty(), "{} should have use cases", info.name);
        }
        for info in &DEPTH_PRESETS {
            assert!(!info.use_cases.is_empty(), "{} should have use cases", info.name);
        }
        for info in &STENCIL_OPERATIONS {
            assert!(!info.use_cases.is_empty(), "{} should have use cases", info.name);
        }
    }

    #[test]
    fn test_depth_format_info_consistency() {
        for info in &DEPTH_FORMATS {
            if info.has_stencil {
                assert!(info.stencil_bits > 0, "{} should have stencil bits", info.name);
            } else {
                assert_eq!(info.stencil_bits, 0, "{} should not have stencil bits", info.name);
            }
            assert!(info.depth_bits >= 24, "{} should have at least 24 depth bits", info.name);
        }
    }

    #[test]
    fn test_stencil_capable_presets_iterator() {
        let count = stencil_capable_presets().count();
        assert_eq!(count, 11);
    }

    #[test]
    fn test_debug_format() {
        let state = DepthStencilStateDescriptor::depth_less();
        let debug = format!("{:?}", state);
        assert!(debug.contains("DepthStencilStateDescriptor"));
    }

    #[test]
    fn test_stencil_face_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(StencilFaceStateDescriptor::default());
        set.insert(StencilFaceStateDescriptor::new().pass_op(wgpu::StencilOperation::Replace));

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_builder_default_trait() {
        let builder1 = DepthStencilStateBuilder::default();
        let builder2 = DepthStencilStateBuilder::new();
        assert_eq!(builder1.build(), builder2.build());
    }
}
