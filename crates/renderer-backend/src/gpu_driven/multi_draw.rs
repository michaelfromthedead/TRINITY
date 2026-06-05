//! Multi-draw indirect wrapper with feature detection and fallback (T-WGPU-P6.7.1).
//!
//! Provides a unified interface for multi-draw indirect commands that automatically
//! falls back to individual draw calls when the `MULTI_DRAW_INDIRECT` feature is
//! unavailable.
//!
//! # Overview
//!
//! GPU-driven rendering often wants to batch many draw calls into a single
//! `multi_draw_indirect` invocation. However, not all hardware supports this:
//!
//! - **Tier 1 (Full)**: `MULTI_DRAW_INDIRECT_COUNT` - GPU-side count buffer
//! - **Tier 2 (Partial)**: `MULTI_DRAW_INDIRECT` - Multiple draws, CPU count
//! - **Tier 3 (Minimal)**: No multi-draw; loop individual calls
//!
//! This module provides transparent fallback so calling code doesn't need to
//! handle feature detection at every call site.
//!
//! # Architecture
//!
//! ```text
//! MultiDrawSupport
//!     |-- new(features) -> Check device capabilities
//!     |-- has_multi_draw() -> bool
//!     `-- has_multi_draw_count() -> bool
//!
//! multi_draw_indirect(pass, support, buffer, offset, count)
//!     |-- if has_multi_draw: pass.multi_draw_indirect(...)
//!     `-- else: for i in 0..count { pass.draw_indirect(...) }
//!
//! multi_draw_indexed_indirect(pass, support, buffer, offset, count)
//!     |-- if has_multi_draw: pass.multi_draw_indexed_indirect(...)
//!     `-- else: for i in 0..count { pass.draw_indexed_indirect(...) }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::gpu_driven::multi_draw::{
//!     MultiDrawSupport, multi_draw_indirect, DRAW_INDIRECT_STRIDE,
//! };
//!
//! // Check once at initialization
//! let support = MultiDrawSupport::new(device.features());
//!
//! // Use transparently in render loop
//! multi_draw_indirect(
//!     &mut render_pass,
//!     &support,
//!     &indirect_buffer,
//!     0,
//!     visible_count,
//! );
//! // This automatically falls back to a loop on devices without multi-draw
//! ```
//!
//! # Performance Considerations
//!
//! The fallback loop incurs overhead proportional to the draw count:
//!
//! | Draw Count | Multi-Draw | Fallback Overhead |
//! |------------|------------|-------------------|
//! | 1          | ~same      | Negligible        |
//! | 10         | 1 call     | ~10x more calls   |
//! | 100        | 1 call     | ~100x more calls  |
//! | 1000       | 1 call     | Consider batching |
//!
//! For very high draw counts on minimal-tier hardware, consider alternative
//! strategies like instancing or mesh merging.
//!
//! # Related Modules
//!
//! - [`indirect_draw`](super::indirect_draw) - Indirect argument structures and tier detection
//! - [`draw_args`](super::draw_args) - Draw argument generation from visibility data
//! - [`stream_compact`](super::stream_compact) - Visibility stream compaction

use log::warn;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::{Buffer, Features, RenderPass};

// =============================================================================
// PERFORMANCE WARNING STATE
// =============================================================================

/// Track whether we've already warned about fallback path usage.
/// We only warn once per session to avoid log spam in hot render loops.
static WARNED_MULTI_DRAW_FALLBACK: AtomicBool = AtomicBool::new(false);
static WARNED_MULTI_DRAW_COUNT_FALLBACK: AtomicBool = AtomicBool::new(false);

/// Emit a one-time warning about multi-draw fallback usage.
#[cold]
fn warn_multi_draw_fallback(count: u32) {
    if !WARNED_MULTI_DRAW_FALLBACK.swap(true, Ordering::Relaxed) {
        warn!(
            "MULTI_DRAW_INDIRECT unsupported: using slow fallback loop for {} draw calls. \
             Consider mesh merging or instancing for better performance.",
            count
        );
    }
}

/// Emit a one-time warning about multi-draw-count fallback usage.
#[cold]
fn warn_multi_draw_count_fallback(fallback_count: u32, max_count: u32) {
    if !WARNED_MULTI_DRAW_COUNT_FALLBACK.swap(true, Ordering::Relaxed) {
        warn!(
            "MULTI_DRAW_INDIRECT_COUNT unsupported: using CPU fallback count ({}/{} max). \
             GPU-driven culling efficiency may be reduced.",
            fallback_count, max_count
        );
    }
}

// =============================================================================
// CONSTANTS
// =============================================================================

/// Stride for DrawIndirectArgs in bytes (4 x u32 = 16 bytes).
///
/// Layout:
/// - vertex_count: u32 (4 bytes, offset 0)
/// - instance_count: u32 (4 bytes, offset 4)
/// - first_vertex: u32 (4 bytes, offset 8)
/// - first_instance: u32 (4 bytes, offset 12)
pub const DRAW_INDIRECT_STRIDE: u64 = 16;

/// Stride for DrawIndexedIndirectArgs in bytes (5 x u32 = 20 bytes).
///
/// Layout:
/// - index_count: u32 (4 bytes, offset 0)
/// - instance_count: u32 (4 bytes, offset 4)
/// - first_index: u32 (4 bytes, offset 8)
/// - base_vertex: i32 (4 bytes, offset 12)
/// - first_instance: u32 (4 bytes, offset 16)
pub const DRAW_INDEXED_INDIRECT_STRIDE: u64 = 20;

// =============================================================================
// MULTI-DRAW SUPPORT DETECTION
// =============================================================================

/// Multi-draw feature support information.
///
/// Caches the result of feature detection to avoid repeated queries.
/// Create once at initialization and reuse throughout the frame.
///
/// # Example
///
/// ```ignore
/// let support = MultiDrawSupport::new(device.features());
///
/// if support.has_multi_draw() {
///     println!("Using native multi-draw");
/// } else {
///     println!("Using fallback loop");
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiDrawSupport {
    /// True if `MULTI_DRAW_INDIRECT` feature is available.
    has_multi_draw: bool,
    /// True if `MULTI_DRAW_INDIRECT_COUNT` feature is available.
    has_multi_draw_count: bool,
}

impl MultiDrawSupport {
    /// Create support info by checking device features.
    ///
    /// # Arguments
    ///
    /// * `features` - Device features from `device.features()` or adapter query.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let support = MultiDrawSupport::new(device.features());
    /// assert!(support.has_multi_draw_count() <= support.has_multi_draw());
    /// ```
    #[inline]
    pub fn new(features: Features) -> Self {
        Self {
            has_multi_draw: features.contains(Features::MULTI_DRAW_INDIRECT),
            has_multi_draw_count: features.contains(Features::MULTI_DRAW_INDIRECT_COUNT),
        }
    }

    /// Create support info from a wgpu device.
    ///
    /// Convenience method that extracts features from the device.
    #[inline]
    pub fn from_device(device: &wgpu::Device) -> Self {
        Self::new(device.features())
    }

    /// Check if native multi-draw indirect is supported.
    ///
    /// When true, `multi_draw_indirect` and `multi_draw_indexed_indirect`
    /// can be called on the render pass directly.
    #[inline]
    pub const fn has_multi_draw(&self) -> bool {
        self.has_multi_draw
    }

    /// Check if multi-draw with GPU count buffer is supported.
    ///
    /// When true, `multi_draw_indirect_count` and `multi_draw_indexed_indirect_count`
    /// can be called, allowing the GPU to determine the draw count.
    ///
    /// Note: `has_multi_draw_count` implies `has_multi_draw`.
    #[inline]
    pub const fn has_multi_draw_count(&self) -> bool {
        self.has_multi_draw_count
    }

    /// Get the capability tier (matches `IndirectTier` semantics).
    ///
    /// - Tier 1 (Full): Both multi-draw and count supported
    /// - Tier 2 (Partial): Multi-draw but not count
    /// - Tier 3 (Minimal): No multi-draw support
    #[inline]
    pub const fn tier(&self) -> u8 {
        if self.has_multi_draw_count {
            1 // Full
        } else if self.has_multi_draw {
            2 // Partial
        } else {
            3 // Minimal
        }
    }

    /// Get a description of the current support level.
    #[inline]
    pub const fn description(&self) -> &'static str {
        if self.has_multi_draw_count {
            "Full (GPU count)"
        } else if self.has_multi_draw {
            "Partial (multi-draw)"
        } else {
            "Minimal (fallback loop)"
        }
    }
}

impl Default for MultiDrawSupport {
    /// Default to minimal support (safest assumption).
    fn default() -> Self {
        Self {
            has_multi_draw: false,
            has_multi_draw_count: false,
        }
    }
}

impl std::fmt::Display for MultiDrawSupport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultiDrawSupport({})", self.description())
    }
}

// =============================================================================
// MULTI-DRAW INDIRECT FUNCTIONS
// =============================================================================

/// Execute multi_draw_indirect with automatic fallback.
///
/// If the device supports `MULTI_DRAW_INDIRECT`, uses the native method.
/// Otherwise, falls back to issuing individual `draw_indirect` calls in a loop.
///
/// # Arguments
///
/// * `render_pass` - The active render pass.
/// * `support` - Pre-checked feature support info.
/// * `indirect_buffer` - Buffer containing [`DrawIndirectArgs`](super::DrawIndirectArgs) entries.
/// * `indirect_offset` - Byte offset into the buffer for the first draw.
/// * `count` - Number of draw commands to execute.
///
/// # Buffer Layout
///
/// The buffer must contain `count` tightly-packed [`DrawIndirectArgs`](super::DrawIndirectArgs)
/// structures starting at `indirect_offset`:
///
/// ```text
/// Buffer Layout:
/// +--------------------+--------------------+--------------------+
/// | DrawIndirectArgs 0 | DrawIndirectArgs 1 | DrawIndirectArgs 2 | ...
/// +--------------------+--------------------+--------------------+
/// ^                    ^
/// indirect_offset      indirect_offset + 16
/// ```
///
/// # Example
///
/// ```ignore
/// use renderer_backend::gpu_driven::multi_draw::{
///     MultiDrawSupport, multi_draw_indirect,
/// };
///
/// let support = MultiDrawSupport::new(device.features());
///
/// // Draw 100 objects from indirect buffer
/// multi_draw_indirect(&mut render_pass, &support, &buffer, 0, 100);
/// ```
#[inline]
pub fn multi_draw_indirect<'a>(
    render_pass: &mut RenderPass<'a>,
    support: &MultiDrawSupport,
    indirect_buffer: &'a Buffer,
    indirect_offset: u64,
    count: u32,
) {
    if count == 0 {
        return;
    }

    if support.has_multi_draw {
        // Native multi-draw: single GPU command
        render_pass.multi_draw_indirect(indirect_buffer, indirect_offset, count);
    } else {
        // Fallback: loop individual draw calls
        // Performance warning: O(count) draw calls instead of O(1)
        warn_multi_draw_fallback(count);
        for i in 0..count {
            let offset = indirect_offset + (i as u64) * DRAW_INDIRECT_STRIDE;
            render_pass.draw_indirect(indirect_buffer, offset);
        }
    }
}

/// Execute multi_draw_indexed_indirect with automatic fallback.
///
/// If the device supports `MULTI_DRAW_INDIRECT`, uses the native method.
/// Otherwise, falls back to issuing individual `draw_indexed_indirect` calls.
///
/// # Arguments
///
/// * `render_pass` - The active render pass.
/// * `support` - Pre-checked feature support info.
/// * `indirect_buffer` - Buffer containing [`DrawIndexedIndirectArgs`](super::DrawIndexedIndirectArgs) entries.
/// * `indirect_offset` - Byte offset into the buffer for the first draw.
/// * `count` - Number of draw commands to execute.
///
/// # Buffer Layout
///
/// The buffer must contain `count` tightly-packed [`DrawIndexedIndirectArgs`](super::DrawIndexedIndirectArgs)
/// structures starting at `indirect_offset`:
///
/// ```text
/// Buffer Layout:
/// +---------------------------+---------------------------+
/// | DrawIndexedIndirectArgs 0 | DrawIndexedIndirectArgs 1 | ...
/// +---------------------------+---------------------------+
/// ^                           ^
/// indirect_offset             indirect_offset + 20
/// ```
///
/// # Example
///
/// ```ignore
/// use renderer_backend::gpu_driven::multi_draw::{
///     MultiDrawSupport, multi_draw_indexed_indirect,
/// };
///
/// let support = MultiDrawSupport::new(device.features());
///
/// // Draw 50 indexed meshes
/// multi_draw_indexed_indirect(&mut render_pass, &support, &buffer, 0, 50);
/// ```
#[inline]
pub fn multi_draw_indexed_indirect<'a>(
    render_pass: &mut RenderPass<'a>,
    support: &MultiDrawSupport,
    indirect_buffer: &'a Buffer,
    indirect_offset: u64,
    count: u32,
) {
    if count == 0 {
        return;
    }

    if support.has_multi_draw {
        // Native multi-draw: single GPU command
        render_pass.multi_draw_indexed_indirect(indirect_buffer, indirect_offset, count);
    } else {
        // Fallback: loop individual draw calls
        // Performance warning: O(count) draw calls instead of O(1)
        warn_multi_draw_fallback(count);
        for i in 0..count {
            let offset = indirect_offset + (i as u64) * DRAW_INDEXED_INDIRECT_STRIDE;
            render_pass.draw_indexed_indirect(indirect_buffer, offset);
        }
    }
}

/// Execute multi_draw_indirect_count with automatic fallback.
///
/// If the device supports `MULTI_DRAW_INDIRECT_COUNT`, uses the native method
/// with GPU-side count. If only `MULTI_DRAW_INDIRECT` is supported, requires
/// `fallback_count` to be provided. Otherwise, falls back to a loop.
///
/// # Arguments
///
/// * `render_pass` - The active render pass.
/// * `support` - Pre-checked feature support info.
/// * `indirect_buffer` - Buffer containing draw argument entries.
/// * `indirect_offset` - Byte offset into the indirect buffer.
/// * `count_buffer` - Buffer containing the draw count (u32).
/// * `count_offset` - Byte offset into the count buffer.
/// * `max_count` - Maximum number of draws (for bounds checking).
/// * `fallback_count` - CPU-side count to use when GPU count unavailable.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::gpu_driven::multi_draw::{
///     MultiDrawSupport, multi_draw_indirect_count,
/// };
///
/// let support = MultiDrawSupport::new(device.features());
///
/// // Try GPU count, fall back to CPU count if needed
/// multi_draw_indirect_count(
///     &mut render_pass,
///     &support,
///     &command_buffer,
///     0,
///     &count_buffer,
///     0,
///     1024,
///     cpu_visible_count, // Used if GPU count unsupported
/// );
/// ```
#[inline]
pub fn multi_draw_indirect_count<'a>(
    render_pass: &mut RenderPass<'a>,
    support: &MultiDrawSupport,
    indirect_buffer: &'a Buffer,
    indirect_offset: u64,
    count_buffer: &'a Buffer,
    count_offset: u64,
    max_count: u32,
    fallback_count: u32,
) {
    if support.has_multi_draw_count {
        // Full support: GPU determines count
        render_pass.multi_draw_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count,
        );
    } else {
        // Partial or minimal: use CPU fallback count instead of GPU count buffer
        warn_multi_draw_count_fallback(fallback_count, max_count);
        let count = fallback_count.min(max_count);
        multi_draw_indirect(render_pass, support, indirect_buffer, indirect_offset, count);
    }
}

/// Execute multi_draw_indexed_indirect_count with automatic fallback.
///
/// Similar to [`multi_draw_indirect_count`] but for indexed drawing.
///
/// # Arguments
///
/// * `render_pass` - The active render pass.
/// * `support` - Pre-checked feature support info.
/// * `indirect_buffer` - Buffer containing indexed draw argument entries.
/// * `indirect_offset` - Byte offset into the indirect buffer.
/// * `count_buffer` - Buffer containing the draw count (u32).
/// * `count_offset` - Byte offset into the count buffer.
/// * `max_count` - Maximum number of draws (for bounds checking).
/// * `fallback_count` - CPU-side count to use when GPU count unavailable.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::gpu_driven::multi_draw::{
///     MultiDrawSupport, multi_draw_indexed_indirect_count,
/// };
///
/// let support = MultiDrawSupport::new(device.features());
///
/// multi_draw_indexed_indirect_count(
///     &mut render_pass,
///     &support,
///     &indexed_command_buffer,
///     0,
///     &count_buffer,
///     0,
///     2048,
///     cpu_count,
/// );
/// ```
#[inline]
pub fn multi_draw_indexed_indirect_count<'a>(
    render_pass: &mut RenderPass<'a>,
    support: &MultiDrawSupport,
    indirect_buffer: &'a Buffer,
    indirect_offset: u64,
    count_buffer: &'a Buffer,
    count_offset: u64,
    max_count: u32,
    fallback_count: u32,
) {
    if support.has_multi_draw_count {
        // Full support: GPU determines count
        render_pass.multi_draw_indexed_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count,
        );
    } else {
        // Partial or minimal: use CPU fallback count instead of GPU count buffer
        warn_multi_draw_count_fallback(fallback_count, max_count);
        let count = fallback_count.min(max_count);
        multi_draw_indexed_indirect(render_pass, support, indirect_buffer, indirect_offset, count);
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Calculate the byte offset for a specific draw command (non-indexed).
///
/// # Example
///
/// ```ignore
/// let offset = draw_indirect_offset(5); // Offset for 6th draw command
/// assert_eq!(offset, 80); // 5 * 16 bytes
/// ```
#[inline]
pub const fn draw_indirect_offset(index: u32) -> u64 {
    (index as u64) * DRAW_INDIRECT_STRIDE
}

/// Calculate the byte offset for a specific draw command (indexed).
///
/// # Example
///
/// ```ignore
/// let offset = draw_indexed_indirect_offset(3); // Offset for 4th draw command
/// assert_eq!(offset, 60); // 3 * 20 bytes
/// ```
#[inline]
pub const fn draw_indexed_indirect_offset(index: u32) -> u64 {
    (index as u64) * DRAW_INDEXED_INDIRECT_STRIDE
}

/// Calculate the minimum buffer size for non-indexed indirect draws.
///
/// # Example
///
/// ```ignore
/// let size = buffer_size_for_draws(100);
/// assert_eq!(size, 1600); // 100 * 16 bytes
/// ```
#[inline]
pub const fn buffer_size_for_draws(count: u32) -> u64 {
    (count as u64) * DRAW_INDIRECT_STRIDE
}

/// Calculate the minimum buffer size for indexed indirect draws.
///
/// # Example
///
/// ```ignore
/// let size = buffer_size_for_indexed_draws(100);
/// assert_eq!(size, 2000); // 100 * 20 bytes
/// ```
#[inline]
pub const fn buffer_size_for_indexed_draws(count: u32) -> u64 {
    (count as u64) * DRAW_INDEXED_INDIRECT_STRIDE
}

// =============================================================================
// TEST HELPERS
// =============================================================================

/// Reset the multi-draw fallback warning state (for testing).
///
/// This allows tests to verify the one-time warning behavior by resetting
/// the atomic flag between test runs.
///
/// # Safety
///
/// This function should only be called from tests. It modifies global state
/// and could cause spurious warnings if called during rendering.

pub fn reset_multi_draw_warning() {
    WARNED_MULTI_DRAW_FALLBACK.store(false, Ordering::Relaxed);
}

/// Reset the multi-draw-count fallback warning state (for testing).
///
/// This allows tests to verify the one-time warning behavior by resetting
/// the atomic flag between test runs.
///
/// # Safety
///
/// This function should only be called from tests. It modifies global state
/// and could cause spurious warnings if called during rendering.

pub fn reset_multi_draw_count_warning() {
    WARNED_MULTI_DRAW_COUNT_FALLBACK.store(false, Ordering::Relaxed);
}

/// Check if the multi-draw fallback warning has been emitted (for testing).

pub fn has_warned_multi_draw_fallback() -> bool {
    WARNED_MULTI_DRAW_FALLBACK.load(Ordering::Relaxed)
}

/// Check if the multi-draw-count fallback warning has been emitted (for testing).

pub fn has_warned_multi_draw_count_fallback() -> bool {
    WARNED_MULTI_DRAW_COUNT_FALLBACK.load(Ordering::Relaxed)
}

/// Trigger the multi-draw fallback warning (for testing).
///
/// This directly calls the warning function without requiring a render pass.

pub fn trigger_multi_draw_warning(count: u32) {
    warn_multi_draw_fallback(count);
}

/// Trigger the multi-draw-count fallback warning (for testing).
///
/// This directly calls the warning function without requiring a render pass.

pub fn trigger_multi_draw_count_warning(fallback_count: u32, max_count: u32) {
    warn_multi_draw_count_fallback(fallback_count, max_count);
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_draw_support_detection_empty() {
        // No features = minimal support
        let support = MultiDrawSupport::new(Features::empty());
        assert!(!support.has_multi_draw());
        assert!(!support.has_multi_draw_count());
        assert_eq!(support.tier(), 3);
        assert_eq!(support.description(), "Minimal (fallback loop)");
    }

    #[test]
    fn test_multi_draw_support_detection_partial() {
        // Only MULTI_DRAW_INDIRECT
        let support = MultiDrawSupport::new(Features::MULTI_DRAW_INDIRECT);
        assert!(support.has_multi_draw());
        assert!(!support.has_multi_draw_count());
        assert_eq!(support.tier(), 2);
        assert_eq!(support.description(), "Partial (multi-draw)");
    }

    #[test]
    fn test_multi_draw_support_detection_full() {
        // Both features
        let features = Features::MULTI_DRAW_INDIRECT | Features::MULTI_DRAW_INDIRECT_COUNT;
        let support = MultiDrawSupport::new(features);
        assert!(support.has_multi_draw());
        assert!(support.has_multi_draw_count());
        assert_eq!(support.tier(), 1);
        assert_eq!(support.description(), "Full (GPU count)");
    }

    #[test]
    fn test_multi_draw_support_default() {
        let support = MultiDrawSupport::default();
        assert!(!support.has_multi_draw());
        assert!(!support.has_multi_draw_count());
        assert_eq!(support.tier(), 3);
    }

    #[test]
    fn test_stride_constants() {
        // Verify stride values match wgpu's expected sizes
        assert_eq!(DRAW_INDIRECT_STRIDE, 16, "DrawIndirectArgs should be 16 bytes");
        assert_eq!(
            DRAW_INDEXED_INDIRECT_STRIDE, 20,
            "DrawIndexedIndirectArgs should be 20 bytes"
        );
    }

    #[test]
    fn test_draw_indirect_offset() {
        assert_eq!(draw_indirect_offset(0), 0);
        assert_eq!(draw_indirect_offset(1), 16);
        assert_eq!(draw_indirect_offset(5), 80);
        assert_eq!(draw_indirect_offset(100), 1600);
    }

    #[test]
    fn test_draw_indexed_indirect_offset() {
        assert_eq!(draw_indexed_indirect_offset(0), 0);
        assert_eq!(draw_indexed_indirect_offset(1), 20);
        assert_eq!(draw_indexed_indirect_offset(5), 100);
        assert_eq!(draw_indexed_indirect_offset(100), 2000);
    }

    #[test]
    fn test_buffer_size_for_draws() {
        assert_eq!(buffer_size_for_draws(0), 0);
        assert_eq!(buffer_size_for_draws(1), 16);
        assert_eq!(buffer_size_for_draws(100), 1600);
        assert_eq!(buffer_size_for_draws(1000), 16000);
    }

    #[test]
    fn test_buffer_size_for_indexed_draws() {
        assert_eq!(buffer_size_for_indexed_draws(0), 0);
        assert_eq!(buffer_size_for_indexed_draws(1), 20);
        assert_eq!(buffer_size_for_indexed_draws(100), 2000);
        assert_eq!(buffer_size_for_indexed_draws(1000), 20000);
    }

    #[test]
    fn test_fallback_count_matches() {
        // Simulate what the fallback loop would do
        let support = MultiDrawSupport::new(Features::empty());
        assert!(!support.has_multi_draw());

        // With fallback, loop count should match requested count
        let count = 10u32;
        let mut loop_count = 0u32;
        for i in 0..count {
            let _offset = (i as u64) * DRAW_INDIRECT_STRIDE;
            loop_count += 1;
        }
        assert_eq!(loop_count, count, "Fallback loop count should match requested count");
    }

    #[test]
    fn test_display_impl() {
        let support = MultiDrawSupport::new(Features::empty());
        let display = format!("{}", support);
        assert!(display.contains("Minimal"));

        let support_full =
            MultiDrawSupport::new(Features::MULTI_DRAW_INDIRECT | Features::MULTI_DRAW_INDIRECT_COUNT);
        let display_full = format!("{}", support_full);
        assert!(display_full.contains("Full"));
    }

    #[test]
    fn test_support_equality() {
        let a = MultiDrawSupport::new(Features::MULTI_DRAW_INDIRECT);
        let b = MultiDrawSupport::new(Features::MULTI_DRAW_INDIRECT);
        let c = MultiDrawSupport::new(Features::empty());

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_support_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(MultiDrawSupport::new(Features::empty()));
        set.insert(MultiDrawSupport::new(Features::MULTI_DRAW_INDIRECT));
        set.insert(MultiDrawSupport::new(
            Features::MULTI_DRAW_INDIRECT | Features::MULTI_DRAW_INDIRECT_COUNT,
        ));

        assert_eq!(set.len(), 3);
    }
}
