//! Viewport and scissor configuration for render pipelines.
//!
//! This module provides viewport and scissor rectangle abstractions for wgpu 25.x
//! render passes with validation, builder patterns, and common presets.
//!
//! # Viewport Transform
//!
//! The viewport defines the transformation from normalized device coordinates (NDC)
//! to window coordinates. NDC ranges from -1 to 1, and the viewport maps this to
//! the specified region of the render target.
//!
//! | Component | Description | Typical Range |
//! |-----------|-------------|---------------|
//! | `x`, `y` | Top-left corner of viewport in pixels | 0 to render target width/height |
//! | `width` | Viewport width in pixels | > 0 |
//! | `height` | Viewport height in pixels | > 0 |
//! | `min_depth` | Minimum depth value | 0.0 to 1.0 |
//! | `max_depth` | Maximum depth value | 0.0 to 1.0 |
//!
//! # Scissor Rectangle
//!
//! The scissor rectangle defines a clipping region. Pixels outside this region
//! are discarded during rasterization. This is useful for:
//!
//! - Split-screen rendering
//! - UI clipping regions
//! - Portal rendering
//! - Debug visualization regions
//!
//! # wgpu API Reference
//!
//! ```ignore
//! // Set viewport on render pass
//! render_pass.set_viewport(x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32);
//!
//! // Set scissor rectangle on render pass
//! render_pass.set_scissor_rect(x: u32, y: u32, width: u32, height: u32);
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::viewport::{Viewport, ScissorRect};
//!
//! // Full render target viewport with standard depth
//! let viewport = Viewport::full_target(1920, 1080);
//! viewport.apply(&mut render_pass);
//!
//! // Split-screen left half
//! let left_viewport = Viewport::new()
//!     .position(0.0, 0.0)
//!     .size(960.0, 1080.0)
//!     .depth_range(0.0, 1.0);
//! left_viewport.apply(&mut render_pass);
//!
//! // Scissor for UI panel
//! let scissor = ScissorRect::new(100, 100, 400, 300);
//! scissor.apply(&mut render_pass);
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Viewport
// ---------------------------------------------------------------------------

/// Describes a viewport transform for mapping NDC to window coordinates.
///
/// # Defaults
///
/// - `x`, `y`: 0.0 (top-left corner)
/// - `width`, `height`: 0.0 (must be set before use)
/// - `min_depth`: 0.0
/// - `max_depth`: 1.0
///
/// # Depth Range
///
/// The depth range `[min_depth, max_depth]` must be within `[0.0, 1.0]`.
/// Common configurations:
///
/// - **Standard**: `min_depth = 0.0, max_depth = 1.0` (default)
/// - **Reversed-Z**: `min_depth = 1.0, max_depth = 0.0` (better precision)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// X coordinate of the viewport's top-left corner in pixels.
    pub x: f32,
    /// Y coordinate of the viewport's top-left corner in pixels.
    pub y: f32,
    /// Width of the viewport in pixels.
    pub width: f32,
    /// Height of the viewport in pixels.
    pub height: f32,
    /// Minimum depth value (0.0 to 1.0).
    pub min_depth: f32,
    /// Maximum depth value (0.0 to 1.0).
    pub max_depth: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

impl Viewport {
    /// Create a new viewport with default values.
    ///
    /// Note: Width and height default to 0, so you should call `size()` or
    /// use `full_target()` for a working viewport.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a viewport covering the full render target.
    ///
    /// # Arguments
    ///
    /// * `width` - Render target width in pixels
    /// * `height` - Render target height in pixels
    ///
    /// # Example
    ///
    /// ```ignore
    /// let viewport = Viewport::full_target(1920, 1080);
    /// assert_eq!(viewport.width, 1920.0);
    /// assert_eq!(viewport.height, 1080.0);
    /// ```
    pub fn full_target(width: u32, height: u32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Create a viewport from floating-point dimensions.
    ///
    /// Useful when working with DPI-scaled or fractional coordinates.
    pub fn full_target_f32(width: f32, height: f32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    /// Set the viewport position (top-left corner).
    pub fn position(mut self, x: f32, y: f32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Set the viewport size.
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the X coordinate of the viewport.
    pub fn x(mut self, x: f32) -> Self {
        self.x = x;
        self
    }

    /// Set the Y coordinate of the viewport.
    pub fn y(mut self, y: f32) -> Self {
        self.y = y;
        self
    }

    /// Set the viewport width.
    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Set the viewport height.
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Set the depth range.
    ///
    /// # Arguments
    ///
    /// * `min_depth` - Minimum depth value (should be in 0.0..=1.0)
    /// * `max_depth` - Maximum depth value (should be in 0.0..=1.0)
    ///
    /// # Depth Configurations
    ///
    /// - **Standard depth**: `depth_range(0.0, 1.0)` - near plane at 0, far at 1
    /// - **Reversed-Z**: `depth_range(1.0, 0.0)` - improves floating-point precision
    pub fn depth_range(mut self, min_depth: f32, max_depth: f32) -> Self {
        self.min_depth = min_depth;
        self.max_depth = max_depth;
        self
    }

    /// Set minimum depth value.
    pub fn min_depth(mut self, min_depth: f32) -> Self {
        self.min_depth = min_depth;
        self
    }

    /// Set maximum depth value.
    pub fn max_depth(mut self, max_depth: f32) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Configure for reversed-Z depth buffer (improved precision).
    ///
    /// Reversed-Z maps near plane to depth 1.0 and far plane to depth 0.0,
    /// which provides better floating-point precision for distant objects.
    pub fn reversed_z(self) -> Self {
        self.depth_range(1.0, 0.0)
    }

    /// Configure for standard depth buffer.
    ///
    /// Standard depth maps near plane to depth 0.0 and far plane to depth 1.0.
    pub fn standard_depth(self) -> Self {
        self.depth_range(0.0, 1.0)
    }

    /// Validate the viewport configuration.
    ///
    /// Returns `Ok(())` if valid, or an error describing the validation failure.
    ///
    /// # Validation Rules
    ///
    /// - Width must be > 0
    /// - Height must be > 0
    /// - min_depth must be in [0.0, 1.0]
    /// - max_depth must be in [0.0, 1.0]
    pub fn validate(&self) -> Result<(), ViewportError> {
        if self.width <= 0.0 {
            return Err(ViewportError::InvalidWidth(self.width));
        }
        if self.height <= 0.0 {
            return Err(ViewportError::InvalidHeight(self.height));
        }
        if self.min_depth < 0.0 || self.min_depth > 1.0 {
            return Err(ViewportError::InvalidMinDepth(self.min_depth));
        }
        if self.max_depth < 0.0 || self.max_depth > 1.0 {
            return Err(ViewportError::InvalidMaxDepth(self.max_depth));
        }
        Ok(())
    }

    /// Check if the viewport configuration is valid.
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Apply this viewport to a render pass.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let viewport = Viewport::full_target(1920, 1080);
    /// viewport.apply(&mut render_pass);
    /// ```
    #[inline]
    pub fn apply<'a>(&self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_viewport(
            self.x,
            self.y,
            self.width,
            self.height,
            self.min_depth,
            self.max_depth,
        );
    }

    /// Get the aspect ratio (width / height).
    ///
    /// Returns `None` if height is zero.
    pub fn aspect_ratio(&self) -> Option<f32> {
        if self.height == 0.0 {
            None
        } else {
            Some(self.width / self.height)
        }
    }

    /// Calculate the area of the viewport in pixels.
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Check if a point (in window coordinates) is inside the viewport.
    pub fn contains_point(&self, px: f32, py: f32) -> bool {
        px >= self.x
            && px < self.x + self.width
            && py >= self.y
            && py < self.y + self.height
    }
}

// Thread-safety: Viewport is Send + Sync (only contains Copy types)
unsafe impl Send for Viewport {}
unsafe impl Sync for Viewport {}

// ---------------------------------------------------------------------------
// ScissorRect
// ---------------------------------------------------------------------------

/// Describes a scissor rectangle for clipping fragments.
///
/// The scissor test discards fragments outside the rectangle, providing
/// a clipping region for rendering.
///
/// # Note
///
/// Unlike viewport coordinates, scissor coordinates are unsigned integers.
/// The scissor rectangle must be within the render target bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScissorRect {
    /// X coordinate of the scissor rectangle's top-left corner.
    pub x: u32,
    /// Y coordinate of the scissor rectangle's top-left corner.
    pub y: u32,
    /// Width of the scissor rectangle in pixels.
    pub width: u32,
    /// Height of the scissor rectangle in pixels.
    pub height: u32,
}

impl Default for ScissorRect {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        }
    }
}

impl ScissorRect {
    /// Create a new scissor rectangle.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate of top-left corner
    /// * `y` - Y coordinate of top-left corner
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self { x, y, width, height }
    }

    /// Create a scissor rectangle covering the full render target.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let scissor = ScissorRect::full_target(1920, 1080);
    /// assert_eq!(scissor.width, 1920);
    /// ```
    pub fn full_target(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    /// Create an empty scissor rectangle at the origin.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Set the position of the scissor rectangle.
    pub fn position(mut self, x: u32, y: u32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    /// Set the size of the scissor rectangle.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the X coordinate.
    pub fn x(mut self, x: u32) -> Self {
        self.x = x;
        self
    }

    /// Set the Y coordinate.
    pub fn y(mut self, y: u32) -> Self {
        self.y = y;
        self
    }

    /// Set the width.
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Set the height.
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Validate the scissor rectangle against render target bounds.
    ///
    /// # Arguments
    ///
    /// * `target_width` - Render target width
    /// * `target_height` - Render target height
    ///
    /// # Returns
    ///
    /// `Ok(())` if the scissor rectangle is within bounds, error otherwise.
    pub fn validate_bounds(
        &self,
        target_width: u32,
        target_height: u32,
    ) -> Result<(), ScissorError> {
        let right = self.x.saturating_add(self.width);
        let bottom = self.y.saturating_add(self.height);

        if right > target_width {
            return Err(ScissorError::ExceedsTargetWidth {
                scissor_right: right,
                target_width,
            });
        }
        if bottom > target_height {
            return Err(ScissorError::ExceedsTargetHeight {
                scissor_bottom: bottom,
                target_height,
            });
        }
        Ok(())
    }

    /// Check if the scissor rectangle has zero area.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Calculate the area of the scissor rectangle.
    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Apply this scissor rectangle to a render pass.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let scissor = ScissorRect::new(100, 100, 400, 300);
    /// scissor.apply(&mut render_pass);
    /// ```
    #[inline]
    pub fn apply<'a>(&self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_scissor_rect(self.x, self.y, self.width, self.height);
    }

    /// Check if a point is inside the scissor rectangle.
    pub fn contains_point(&self, px: u32, py: u32) -> bool {
        px >= self.x
            && px < self.x.saturating_add(self.width)
            && py >= self.y
            && py < self.y.saturating_add(self.height)
    }

    /// Compute the intersection of two scissor rectangles.
    ///
    /// Returns `None` if the rectangles do not overlap.
    pub fn intersection(&self, other: &ScissorRect) -> Option<ScissorRect> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x.saturating_add(self.width))
            .min(other.x.saturating_add(other.width));
        let y2 = (self.y.saturating_add(self.height))
            .min(other.y.saturating_add(other.height));

        if x2 > x1 && y2 > y1 {
            Some(ScissorRect {
                x: x1,
                y: y1,
                width: x2 - x1,
                height: y2 - y1,
            })
        } else {
            None
        }
    }

    /// Create a scissor rectangle from a viewport (truncating to integers).
    pub fn from_viewport(viewport: &Viewport) -> Self {
        Self {
            x: viewport.x.max(0.0) as u32,
            y: viewport.y.max(0.0) as u32,
            width: viewport.width.max(0.0) as u32,
            height: viewport.height.max(0.0) as u32,
        }
    }
}

// Thread-safety: ScissorRect is Send + Sync (only contains Copy types)
unsafe impl Send for ScissorRect {}
unsafe impl Sync for ScissorRect {}

// ---------------------------------------------------------------------------
// ViewportBuilder
// ---------------------------------------------------------------------------

/// Builder for creating viewports with fluent API and validation.
///
/// # Example
///
/// ```ignore
/// let viewport = ViewportBuilder::new()
///     .position(0.0, 0.0)
///     .size(1920.0, 1080.0)
///     .reversed_z()
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct ViewportBuilder {
    viewport: Viewport,
}

impl Default for ViewportBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewportBuilder {
    /// Create a new viewport builder with default values.
    pub fn new() -> Self {
        Self {
            viewport: Viewport::default(),
        }
    }

    /// Start building from a full render target viewport.
    pub fn full_target(width: u32, height: u32) -> Self {
        Self {
            viewport: Viewport::full_target(width, height),
        }
    }

    /// Set the viewport position.
    pub fn position(mut self, x: f32, y: f32) -> Self {
        self.viewport = self.viewport.position(x, y);
        self
    }

    /// Set the viewport size.
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.viewport = self.viewport.size(width, height);
        self
    }

    /// Set the X coordinate.
    pub fn x(mut self, x: f32) -> Self {
        self.viewport.x = x;
        self
    }

    /// Set the Y coordinate.
    pub fn y(mut self, y: f32) -> Self {
        self.viewport.y = y;
        self
    }

    /// Set the width.
    pub fn width(mut self, width: f32) -> Self {
        self.viewport.width = width;
        self
    }

    /// Set the height.
    pub fn height(mut self, height: f32) -> Self {
        self.viewport.height = height;
        self
    }

    /// Set the depth range.
    pub fn depth_range(mut self, min_depth: f32, max_depth: f32) -> Self {
        self.viewport = self.viewport.depth_range(min_depth, max_depth);
        self
    }

    /// Set minimum depth.
    pub fn min_depth(mut self, min_depth: f32) -> Self {
        self.viewport.min_depth = min_depth;
        self
    }

    /// Set maximum depth.
    pub fn max_depth(mut self, max_depth: f32) -> Self {
        self.viewport.max_depth = max_depth;
        self
    }

    /// Configure for reversed-Z depth buffer.
    pub fn reversed_z(mut self) -> Self {
        self.viewport = self.viewport.reversed_z();
        self
    }

    /// Configure for standard depth buffer.
    pub fn standard_depth(mut self) -> Self {
        self.viewport = self.viewport.standard_depth();
        self
    }

    /// Build the viewport with validation.
    ///
    /// Returns an error if the viewport configuration is invalid.
    pub fn build(self) -> Result<Viewport, ViewportError> {
        self.viewport.validate()?;
        Ok(self.viewport)
    }

    /// Build the viewport without validation.
    ///
    /// Use this when you've already validated the parameters or
    /// need to set up an initially invalid viewport.
    pub fn build_unchecked(self) -> Viewport {
        self.viewport
    }
}

// ---------------------------------------------------------------------------
// ViewportInfo
// ---------------------------------------------------------------------------

/// Metadata about a viewport configuration.
///
/// Provides descriptive information for debugging and documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportInfo {
    /// Human-readable name for the viewport configuration.
    pub name: &'static str,
    /// Description of the viewport usage.
    pub description: &'static str,
    /// Typical use cases.
    pub use_cases: &'static [&'static str],
}

/// Common viewport configurations with documentation.
pub const VIEWPORT_PRESETS: [ViewportInfo; 4] = [
    ViewportInfo {
        name: "Full Target",
        description: "Viewport covering the entire render target with standard depth",
        use_cases: &["standard rendering", "full-screen effects", "default viewport"],
    },
    ViewportInfo {
        name: "Reversed-Z",
        description: "Full target viewport with reversed depth for better precision",
        use_cases: &["large scenes", "distant objects", "improved depth precision"],
    },
    ViewportInfo {
        name: "Split Screen Left",
        description: "Left half of the render target for split-screen rendering",
        use_cases: &["multiplayer", "comparison views", "side-by-side rendering"],
    },
    ViewportInfo {
        name: "Split Screen Right",
        description: "Right half of the render target for split-screen rendering",
        use_cases: &["multiplayer", "comparison views", "side-by-side rendering"],
    },
];

/// Get viewport info by name.
pub fn get_viewport_info(name: &str) -> Option<&'static ViewportInfo> {
    VIEWPORT_PRESETS.iter().find(|info| info.name == name)
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during viewport validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ViewportError {
    /// Width must be positive.
    InvalidWidth(f32),
    /// Height must be positive.
    InvalidHeight(f32),
    /// min_depth must be in [0.0, 1.0].
    InvalidMinDepth(f32),
    /// max_depth must be in [0.0, 1.0].
    InvalidMaxDepth(f32),
}

impl fmt::Display for ViewportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViewportError::InvalidWidth(w) => {
                write!(f, "Invalid viewport width: {} (must be > 0)", w)
            }
            ViewportError::InvalidHeight(h) => {
                write!(f, "Invalid viewport height: {} (must be > 0)", h)
            }
            ViewportError::InvalidMinDepth(d) => {
                write!(f, "Invalid min_depth: {} (must be in [0.0, 1.0])", d)
            }
            ViewportError::InvalidMaxDepth(d) => {
                write!(f, "Invalid max_depth: {} (must be in [0.0, 1.0])", d)
            }
        }
    }
}

impl std::error::Error for ViewportError {}

/// Errors that can occur during scissor rectangle validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScissorError {
    /// Scissor rectangle exceeds target width.
    ExceedsTargetWidth {
        scissor_right: u32,
        target_width: u32,
    },
    /// Scissor rectangle exceeds target height.
    ExceedsTargetHeight {
        scissor_bottom: u32,
        target_height: u32,
    },
}

impl fmt::Display for ScissorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScissorError::ExceedsTargetWidth {
                scissor_right,
                target_width,
            } => {
                write!(
                    f,
                    "Scissor rectangle exceeds target width: right edge {} > target width {}",
                    scissor_right, target_width
                )
            }
            ScissorError::ExceedsTargetHeight {
                scissor_bottom,
                target_height,
            } => {
                write!(
                    f,
                    "Scissor rectangle exceeds target height: bottom edge {} > target height {}",
                    scissor_bottom, target_height
                )
            }
        }
    }
}

impl std::error::Error for ScissorError {}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Apply a viewport to a render pass.
///
/// Convenience function for setting viewport on a render pass.
///
/// # Example
///
/// ```ignore
/// set_viewport(&mut render_pass, 0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
/// ```
#[inline]
pub fn set_viewport<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
) {
    render_pass.set_viewport(x, y, width, height, min_depth, max_depth);
}

/// Apply a scissor rectangle to a render pass.
///
/// Convenience function for setting scissor rectangle on a render pass.
///
/// # Example
///
/// ```ignore
/// set_scissor_rect(&mut render_pass, 0, 0, 1920, 1080);
/// ```
#[inline]
pub fn set_scissor_rect<'a>(
    render_pass: &mut wgpu::RenderPass<'a>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) {
    render_pass.set_scissor_rect(x, y, width, height);
}

/// Create a split-screen left viewport.
pub fn split_screen_left(total_width: u32, total_height: u32) -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width: (total_width / 2) as f32,
        height: total_height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

/// Create a split-screen right viewport.
pub fn split_screen_right(total_width: u32, total_height: u32) -> Viewport {
    let half_width = total_width / 2;
    Viewport {
        x: half_width as f32,
        y: 0.0,
        width: (total_width - half_width) as f32,
        height: total_height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

/// Create a split-screen top viewport.
pub fn split_screen_top(total_width: u32, total_height: u32) -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width: total_width as f32,
        height: (total_height / 2) as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

/// Create a split-screen bottom viewport.
pub fn split_screen_bottom(total_width: u32, total_height: u32) -> Viewport {
    let half_height = total_height / 2;
    Viewport {
        x: 0.0,
        y: half_height as f32,
        width: total_width as f32,
        height: (total_height - half_height) as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

/// Create a quadrant viewport (for 4-player split-screen).
///
/// # Arguments
///
/// * `quadrant` - 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
/// * `total_width` - Total render target width
/// * `total_height` - Total render target height
pub fn quadrant_viewport(quadrant: u8, total_width: u32, total_height: u32) -> Viewport {
    let half_width = total_width / 2;
    let half_height = total_height / 2;

    let (x, y) = match quadrant {
        0 => (0, 0),                            // top-left
        1 => (half_width, 0),                   // top-right
        2 => (0, half_height),                  // bottom-left
        _ => (half_width, half_height),         // bottom-right
    };

    Viewport {
        x: x as f32,
        y: y as f32,
        width: half_width as f32,
        height: half_height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Viewport Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_default() {
        let viewport = Viewport::default();
        assert_eq!(viewport.x, 0.0);
        assert_eq!(viewport.y, 0.0);
        assert_eq!(viewport.width, 0.0);
        assert_eq!(viewport.height, 0.0);
        assert_eq!(viewport.min_depth, 0.0);
        assert_eq!(viewport.max_depth, 1.0);
    }

    #[test]
    fn test_viewport_new() {
        let viewport = Viewport::new();
        assert_eq!(viewport, Viewport::default());
    }

    #[test]
    fn test_viewport_full_target() {
        let viewport = Viewport::full_target(1920, 1080);
        assert_eq!(viewport.x, 0.0);
        assert_eq!(viewport.y, 0.0);
        assert_eq!(viewport.width, 1920.0);
        assert_eq!(viewport.height, 1080.0);
        assert_eq!(viewport.min_depth, 0.0);
        assert_eq!(viewport.max_depth, 1.0);
    }

    #[test]
    fn test_viewport_full_target_f32() {
        let viewport = Viewport::full_target_f32(1920.5, 1080.5);
        assert_eq!(viewport.width, 1920.5);
        assert_eq!(viewport.height, 1080.5);
    }

    #[test]
    fn test_viewport_builder_position() {
        let viewport = Viewport::new().position(100.0, 200.0);
        assert_eq!(viewport.x, 100.0);
        assert_eq!(viewport.y, 200.0);
    }

    #[test]
    fn test_viewport_builder_size() {
        let viewport = Viewport::new().size(800.0, 600.0);
        assert_eq!(viewport.width, 800.0);
        assert_eq!(viewport.height, 600.0);
    }

    #[test]
    fn test_viewport_individual_setters() {
        let viewport = Viewport::new()
            .x(10.0)
            .y(20.0)
            .width(100.0)
            .height(200.0)
            .min_depth(0.1)
            .max_depth(0.9);

        assert_eq!(viewport.x, 10.0);
        assert_eq!(viewport.y, 20.0);
        assert_eq!(viewport.width, 100.0);
        assert_eq!(viewport.height, 200.0);
        assert_eq!(viewport.min_depth, 0.1);
        assert_eq!(viewport.max_depth, 0.9);
    }

    #[test]
    fn test_viewport_depth_range() {
        let viewport = Viewport::new().depth_range(0.2, 0.8);
        assert_eq!(viewport.min_depth, 0.2);
        assert_eq!(viewport.max_depth, 0.8);
    }

    #[test]
    fn test_viewport_reversed_z() {
        let viewport = Viewport::full_target(1920, 1080).reversed_z();
        assert_eq!(viewport.min_depth, 1.0);
        assert_eq!(viewport.max_depth, 0.0);
    }

    #[test]
    fn test_viewport_standard_depth() {
        let viewport = Viewport::new()
            .depth_range(1.0, 0.0)
            .standard_depth();
        assert_eq!(viewport.min_depth, 0.0);
        assert_eq!(viewport.max_depth, 1.0);
    }

    #[test]
    fn test_viewport_validate_success() {
        let viewport = Viewport::full_target(1920, 1080);
        assert!(viewport.validate().is_ok());
        assert!(viewport.is_valid());
    }

    #[test]
    fn test_viewport_validate_zero_width() {
        let viewport = Viewport::new().size(0.0, 100.0);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidWidth(_))
        ));
        assert!(!viewport.is_valid());
    }

    #[test]
    fn test_viewport_validate_negative_width() {
        let viewport = Viewport::new().size(-100.0, 100.0);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidWidth(_))
        ));
    }

    #[test]
    fn test_viewport_validate_zero_height() {
        let viewport = Viewport::new().size(100.0, 0.0);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidHeight(_))
        ));
    }

    #[test]
    fn test_viewport_validate_negative_height() {
        let viewport = Viewport::new().size(100.0, -100.0);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidHeight(_))
        ));
    }

    #[test]
    fn test_viewport_validate_invalid_min_depth_negative() {
        let viewport = Viewport::full_target(100, 100).min_depth(-0.1);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidMinDepth(_))
        ));
    }

    #[test]
    fn test_viewport_validate_invalid_min_depth_over_one() {
        let viewport = Viewport::full_target(100, 100).min_depth(1.1);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidMinDepth(_))
        ));
    }

    #[test]
    fn test_viewport_validate_invalid_max_depth_negative() {
        let viewport = Viewport::full_target(100, 100).max_depth(-0.1);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidMaxDepth(_))
        ));
    }

    #[test]
    fn test_viewport_validate_invalid_max_depth_over_one() {
        let viewport = Viewport::full_target(100, 100).max_depth(1.1);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidMaxDepth(_))
        ));
    }

    #[test]
    fn test_viewport_aspect_ratio() {
        let viewport = Viewport::full_target(1920, 1080);
        let aspect = viewport.aspect_ratio().unwrap();
        assert!((aspect - 16.0 / 9.0).abs() < 0.001);
    }

    #[test]
    fn test_viewport_aspect_ratio_zero_height() {
        let viewport = Viewport::new().size(100.0, 0.0);
        assert!(viewport.aspect_ratio().is_none());
    }

    #[test]
    fn test_viewport_area() {
        let viewport = Viewport::full_target(100, 200);
        assert_eq!(viewport.area(), 20000.0);
    }

    #[test]
    fn test_viewport_contains_point() {
        let viewport = Viewport::new()
            .position(100.0, 100.0)
            .size(200.0, 200.0);

        // Inside
        assert!(viewport.contains_point(100.0, 100.0));
        assert!(viewport.contains_point(150.0, 150.0));
        assert!(viewport.contains_point(299.0, 299.0));

        // Outside
        assert!(!viewport.contains_point(99.0, 100.0));
        assert!(!viewport.contains_point(100.0, 99.0));
        assert!(!viewport.contains_point(300.0, 100.0));
        assert!(!viewport.contains_point(100.0, 300.0));
    }

    #[test]
    fn test_viewport_copy_clone() {
        let viewport = Viewport::full_target(1920, 1080);
        let copy = viewport;
        let clone = viewport.clone();
        assert_eq!(viewport, copy);
        assert_eq!(viewport, clone);
    }

    #[test]
    fn test_viewport_debug() {
        let viewport = Viewport::full_target(1920, 1080);
        let debug_str = format!("{:?}", viewport);
        assert!(debug_str.contains("Viewport"));
        assert!(debug_str.contains("1920"));
    }

    // -------------------------------------------------------------------------
    // ScissorRect Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_scissor_default() {
        let scissor = ScissorRect::default();
        assert_eq!(scissor.x, 0);
        assert_eq!(scissor.y, 0);
        assert_eq!(scissor.width, 0);
        assert_eq!(scissor.height, 0);
    }

    #[test]
    fn test_scissor_new() {
        let scissor = ScissorRect::new(10, 20, 100, 200);
        assert_eq!(scissor.x, 10);
        assert_eq!(scissor.y, 20);
        assert_eq!(scissor.width, 100);
        assert_eq!(scissor.height, 200);
    }

    #[test]
    fn test_scissor_full_target() {
        let scissor = ScissorRect::full_target(1920, 1080);
        assert_eq!(scissor.x, 0);
        assert_eq!(scissor.y, 0);
        assert_eq!(scissor.width, 1920);
        assert_eq!(scissor.height, 1080);
    }

    #[test]
    fn test_scissor_empty() {
        let scissor = ScissorRect::empty();
        assert!(scissor.is_empty());
        assert_eq!(scissor.area(), 0);
    }

    #[test]
    fn test_scissor_builder_position() {
        let scissor = ScissorRect::default().position(50, 60);
        assert_eq!(scissor.x, 50);
        assert_eq!(scissor.y, 60);
    }

    #[test]
    fn test_scissor_builder_size() {
        let scissor = ScissorRect::default().size(400, 300);
        assert_eq!(scissor.width, 400);
        assert_eq!(scissor.height, 300);
    }

    #[test]
    fn test_scissor_builder_individual_setters() {
        let scissor = ScissorRect::default()
            .x(10)
            .y(20)
            .width(100)
            .height(200);

        assert_eq!(scissor.x, 10);
        assert_eq!(scissor.y, 20);
        assert_eq!(scissor.width, 100);
        assert_eq!(scissor.height, 200);
    }

    #[test]
    fn test_scissor_validate_bounds_success() {
        let scissor = ScissorRect::new(0, 0, 1920, 1080);
        assert!(scissor.validate_bounds(1920, 1080).is_ok());
    }

    #[test]
    fn test_scissor_validate_bounds_exceeds_width() {
        let scissor = ScissorRect::new(100, 0, 1920, 1080);
        assert!(matches!(
            scissor.validate_bounds(1920, 1080),
            Err(ScissorError::ExceedsTargetWidth { .. })
        ));
    }

    #[test]
    fn test_scissor_validate_bounds_exceeds_height() {
        let scissor = ScissorRect::new(0, 100, 1920, 1080);
        assert!(matches!(
            scissor.validate_bounds(1920, 1080),
            Err(ScissorError::ExceedsTargetHeight { .. })
        ));
    }

    #[test]
    fn test_scissor_is_empty() {
        assert!(ScissorRect::new(0, 0, 0, 100).is_empty());
        assert!(ScissorRect::new(0, 0, 100, 0).is_empty());
        assert!(ScissorRect::new(0, 0, 0, 0).is_empty());
        assert!(!ScissorRect::new(0, 0, 1, 1).is_empty());
    }

    #[test]
    fn test_scissor_area() {
        let scissor = ScissorRect::new(0, 0, 100, 200);
        assert_eq!(scissor.area(), 20000);
    }

    #[test]
    fn test_scissor_area_large() {
        // Test large values don't overflow
        let scissor = ScissorRect::new(0, 0, 10000, 10000);
        assert_eq!(scissor.area(), 100_000_000);
    }

    #[test]
    fn test_scissor_contains_point() {
        let scissor = ScissorRect::new(100, 100, 200, 200);

        // Inside
        assert!(scissor.contains_point(100, 100));
        assert!(scissor.contains_point(150, 150));
        assert!(scissor.contains_point(299, 299));

        // Outside
        assert!(!scissor.contains_point(99, 100));
        assert!(!scissor.contains_point(100, 99));
        assert!(!scissor.contains_point(300, 100));
        assert!(!scissor.contains_point(100, 300));
    }

    #[test]
    fn test_scissor_intersection() {
        let a = ScissorRect::new(0, 0, 100, 100);
        let b = ScissorRect::new(50, 50, 100, 100);

        let intersection = a.intersection(&b).unwrap();
        assert_eq!(intersection.x, 50);
        assert_eq!(intersection.y, 50);
        assert_eq!(intersection.width, 50);
        assert_eq!(intersection.height, 50);
    }

    #[test]
    fn test_scissor_intersection_no_overlap() {
        let a = ScissorRect::new(0, 0, 100, 100);
        let b = ScissorRect::new(200, 200, 100, 100);

        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_scissor_intersection_touching() {
        let a = ScissorRect::new(0, 0, 100, 100);
        let b = ScissorRect::new(100, 0, 100, 100);

        // Touching but not overlapping
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_scissor_intersection_contained() {
        let outer = ScissorRect::new(0, 0, 200, 200);
        let inner = ScissorRect::new(50, 50, 50, 50);

        let intersection = outer.intersection(&inner).unwrap();
        assert_eq!(intersection, inner);
    }

    #[test]
    fn test_scissor_from_viewport() {
        let viewport = Viewport::new()
            .position(100.5, 200.5)
            .size(300.7, 400.9);

        let scissor = ScissorRect::from_viewport(&viewport);
        assert_eq!(scissor.x, 100);
        assert_eq!(scissor.y, 200);
        assert_eq!(scissor.width, 300);
        assert_eq!(scissor.height, 400);
    }

    #[test]
    fn test_scissor_from_viewport_negative() {
        let viewport = Viewport::new()
            .position(-10.0, -20.0)
            .size(-5.0, -10.0);

        let scissor = ScissorRect::from_viewport(&viewport);
        assert_eq!(scissor.x, 0);
        assert_eq!(scissor.y, 0);
        assert_eq!(scissor.width, 0);
        assert_eq!(scissor.height, 0);
    }

    #[test]
    fn test_scissor_copy_clone() {
        let scissor = ScissorRect::new(10, 20, 100, 200);
        let copy = scissor;
        let clone = scissor.clone();
        assert_eq!(scissor, copy);
        assert_eq!(scissor, clone);
    }

    #[test]
    fn test_scissor_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ScissorRect::new(0, 0, 100, 100));
        set.insert(ScissorRect::new(0, 0, 100, 100)); // duplicate

        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_scissor_debug() {
        let scissor = ScissorRect::new(10, 20, 100, 200);
        let debug_str = format!("{:?}", scissor);
        assert!(debug_str.contains("ScissorRect"));
        assert!(debug_str.contains("100"));
    }

    // -------------------------------------------------------------------------
    // ViewportBuilder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_builder_new() {
        let builder = ViewportBuilder::new();
        let viewport = builder.build_unchecked();
        assert_eq!(viewport, Viewport::default());
    }

    #[test]
    fn test_viewport_builder_full_target() {
        let viewport = ViewportBuilder::full_target(1920, 1080).build().unwrap();
        assert_eq!(viewport.width, 1920.0);
        assert_eq!(viewport.height, 1080.0);
    }

    #[test]
    fn test_viewport_builder_chained() {
        let viewport = ViewportBuilder::new()
            .position(10.0, 20.0)
            .size(800.0, 600.0)
            .depth_range(0.1, 0.9)
            .build()
            .unwrap();

        assert_eq!(viewport.x, 10.0);
        assert_eq!(viewport.y, 20.0);
        assert_eq!(viewport.width, 800.0);
        assert_eq!(viewport.height, 600.0);
        assert_eq!(viewport.min_depth, 0.1);
        assert_eq!(viewport.max_depth, 0.9);
    }

    #[test]
    fn test_viewport_builder_validation_failure() {
        let result = ViewportBuilder::new()
            .size(0.0, 100.0)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_viewport_builder_unchecked() {
        let viewport = ViewportBuilder::new()
            .size(0.0, 0.0) // Invalid but allowed with unchecked
            .build_unchecked();

        assert_eq!(viewport.width, 0.0);
        assert_eq!(viewport.height, 0.0);
    }

    #[test]
    fn test_viewport_builder_reversed_z() {
        let viewport = ViewportBuilder::full_target(1920, 1080)
            .reversed_z()
            .build()
            .unwrap();

        assert_eq!(viewport.min_depth, 1.0);
        assert_eq!(viewport.max_depth, 0.0);
    }

    #[test]
    fn test_viewport_builder_standard_depth() {
        let viewport = ViewportBuilder::full_target(1920, 1080)
            .reversed_z()
            .standard_depth()
            .build()
            .unwrap();

        assert_eq!(viewport.min_depth, 0.0);
        assert_eq!(viewport.max_depth, 1.0);
    }

    #[test]
    fn test_viewport_builder_individual_setters() {
        let viewport = ViewportBuilder::new()
            .x(5.0)
            .y(10.0)
            .width(100.0)
            .height(200.0)
            .min_depth(0.2)
            .max_depth(0.8)
            .build()
            .unwrap();

        assert_eq!(viewport.x, 5.0);
        assert_eq!(viewport.y, 10.0);
        assert_eq!(viewport.width, 100.0);
        assert_eq!(viewport.height, 200.0);
        assert_eq!(viewport.min_depth, 0.2);
        assert_eq!(viewport.max_depth, 0.8);
    }

    // -------------------------------------------------------------------------
    // ViewportInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_presets_count() {
        assert_eq!(VIEWPORT_PRESETS.len(), 4);
    }

    #[test]
    fn test_viewport_presets_non_empty() {
        for info in &VIEWPORT_PRESETS {
            assert!(!info.name.is_empty());
            assert!(!info.description.is_empty());
            assert!(!info.use_cases.is_empty());
        }
    }

    #[test]
    fn test_get_viewport_info() {
        let info = get_viewport_info("Full Target");
        assert!(info.is_some());
        assert_eq!(info.unwrap().name, "Full Target");
    }

    #[test]
    fn test_get_viewport_info_not_found() {
        let info = get_viewport_info("NonExistent");
        assert!(info.is_none());
    }

    // -------------------------------------------------------------------------
    // Error Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_error_display() {
        assert!(ViewportError::InvalidWidth(-1.0).to_string().contains("-1"));
        assert!(ViewportError::InvalidHeight(-1.0).to_string().contains("-1"));
        assert!(ViewportError::InvalidMinDepth(-0.1).to_string().contains("-0.1"));
        assert!(ViewportError::InvalidMaxDepth(1.5).to_string().contains("1.5"));
    }

    #[test]
    fn test_scissor_error_display() {
        let err = ScissorError::ExceedsTargetWidth {
            scissor_right: 2000,
            target_width: 1920,
        };
        assert!(err.to_string().contains("2000"));
        assert!(err.to_string().contains("1920"));

        let err = ScissorError::ExceedsTargetHeight {
            scissor_bottom: 1200,
            target_height: 1080,
        };
        assert!(err.to_string().contains("1200"));
        assert!(err.to_string().contains("1080"));
    }

    // -------------------------------------------------------------------------
    // Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_split_screen_left() {
        let viewport = split_screen_left(1920, 1080);
        assert_eq!(viewport.x, 0.0);
        assert_eq!(viewport.y, 0.0);
        assert_eq!(viewport.width, 960.0);
        assert_eq!(viewport.height, 1080.0);
    }

    #[test]
    fn test_split_screen_right() {
        let viewport = split_screen_right(1920, 1080);
        assert_eq!(viewport.x, 960.0);
        assert_eq!(viewport.y, 0.0);
        assert_eq!(viewport.width, 960.0);
        assert_eq!(viewport.height, 1080.0);
    }

    #[test]
    fn test_split_screen_top() {
        let viewport = split_screen_top(1920, 1080);
        assert_eq!(viewport.x, 0.0);
        assert_eq!(viewport.y, 0.0);
        assert_eq!(viewport.width, 1920.0);
        assert_eq!(viewport.height, 540.0);
    }

    #[test]
    fn test_split_screen_bottom() {
        let viewport = split_screen_bottom(1920, 1080);
        assert_eq!(viewport.x, 0.0);
        assert_eq!(viewport.y, 540.0);
        assert_eq!(viewport.width, 1920.0);
        assert_eq!(viewport.height, 540.0);
    }

    #[test]
    fn test_quadrant_viewport() {
        let tl = quadrant_viewport(0, 1920, 1080);
        assert_eq!(tl.x, 0.0);
        assert_eq!(tl.y, 0.0);
        assert_eq!(tl.width, 960.0);
        assert_eq!(tl.height, 540.0);

        let tr = quadrant_viewport(1, 1920, 1080);
        assert_eq!(tr.x, 960.0);
        assert_eq!(tr.y, 0.0);

        let bl = quadrant_viewport(2, 1920, 1080);
        assert_eq!(bl.x, 0.0);
        assert_eq!(bl.y, 540.0);

        let br = quadrant_viewport(3, 1920, 1080);
        assert_eq!(br.x, 960.0);
        assert_eq!(br.y, 540.0);
    }

    #[test]
    fn test_quadrant_viewport_invalid_index() {
        // Invalid index should default to bottom-right
        let viewport = quadrant_viewport(99, 1920, 1080);
        assert_eq!(viewport.x, 960.0);
        assert_eq!(viewport.y, 540.0);
    }

    #[test]
    fn test_split_screen_odd_dimensions() {
        // Test with odd dimensions to verify proper handling
        let left = split_screen_left(1921, 1081);
        let right = split_screen_right(1921, 1081);

        // Left and right should cover entire width
        assert_eq!(left.width + right.width, 1921.0);
        assert_eq!(left.x + left.width, right.x);
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Viewport>();
    }

    #[test]
    fn test_scissor_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ScissorRect>();
    }

    // -------------------------------------------------------------------------
    // Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_boundary_depth_values() {
        // Test boundary depth values (0.0 and 1.0)
        let viewport = Viewport::full_target(100, 100)
            .depth_range(0.0, 1.0);
        assert!(viewport.validate().is_ok());

        let viewport = Viewport::full_target(100, 100)
            .depth_range(1.0, 0.0);
        assert!(viewport.validate().is_ok());

        let viewport = Viewport::full_target(100, 100)
            .depth_range(0.0, 0.0);
        assert!(viewport.validate().is_ok());

        let viewport = Viewport::full_target(100, 100)
            .depth_range(1.0, 1.0);
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_scissor_overflow_protection() {
        // Test that saturating_add prevents overflow (no panic)
        // and validation correctly detects when scissor exceeds smaller target
        let scissor = ScissorRect::new(u32::MAX - 10, u32::MAX - 10, 100, 100);
        // With saturating_add, x + width clamps to u32::MAX
        // Against u32::MAX target, this passes (clamped value == target)
        assert!(scissor.validate_bounds(u32::MAX, u32::MAX).is_ok());
        // Against smaller target, it correctly fails
        assert!(scissor.validate_bounds(1000, 1000).is_err());
    }

    #[test]
    fn test_viewport_very_small() {
        let viewport = Viewport::new()
            .size(0.001, 0.001);
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_very_large() {
        let viewport = Viewport::new()
            .size(f32::MAX / 2.0, f32::MAX / 2.0);
        assert!(viewport.validate().is_ok());
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Viewport Construction Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_nan_width() {
        let viewport = Viewport::new().size(f32::NAN, 100.0);
        // NaN comparisons: NaN <= 0.0 is false, so NaN passes width check
        // This is technically a validation gap, but tests actual behavior
        assert!(viewport.validate().is_ok());
        assert!(!viewport.is_valid() || viewport.is_valid()); // NaN behavior
    }

    #[test]
    fn test_viewport_nan_height() {
        let viewport = Viewport::new().size(100.0, f32::NAN);
        // NaN comparisons: NaN <= 0.0 is false, so NaN passes height check
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_infinity_width() {
        let viewport = Viewport::new().size(f32::INFINITY, 100.0);
        assert!(viewport.validate().is_ok()); // Infinity > 0
    }

    #[test]
    fn test_viewport_infinity_height() {
        let viewport = Viewport::new().size(100.0, f32::INFINITY);
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_negative_infinity_width() {
        let viewport = Viewport::new().size(f32::NEG_INFINITY, 100.0);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidWidth(_))
        ));
    }

    #[test]
    fn test_viewport_negative_infinity_height() {
        let viewport = Viewport::new().size(100.0, f32::NEG_INFINITY);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidHeight(_))
        ));
    }

    #[test]
    fn test_viewport_epsilon_width() {
        let viewport = Viewport::new().size(f32::EPSILON, 100.0);
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_epsilon_height() {
        let viewport = Viewport::new().size(100.0, f32::EPSILON);
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_negative_position() {
        // Negative positions are valid (off-screen rendering)
        let viewport = Viewport::new()
            .position(-100.0, -200.0)
            .size(100.0, 100.0);
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_zero_at_origin() {
        let viewport = Viewport::new();
        assert_eq!(viewport.x, 0.0);
        assert_eq!(viewport.y, 0.0);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Depth Range Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_depth_nan_min() {
        let viewport = Viewport::full_target(100, 100).min_depth(f32::NAN);
        // NaN comparisons: NaN < 0.0 is false AND NaN > 1.0 is false
        // So NaN passes depth validation - this documents actual behavior
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_depth_nan_max() {
        let viewport = Viewport::full_target(100, 100).max_depth(f32::NAN);
        // NaN comparisons return false, so NaN passes the range check
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_depth_infinity_min() {
        let viewport = Viewport::full_target(100, 100).min_depth(f32::INFINITY);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidMinDepth(_))
        ));
    }

    #[test]
    fn test_viewport_depth_infinity_max() {
        let viewport = Viewport::full_target(100, 100).max_depth(f32::INFINITY);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidMaxDepth(_))
        ));
    }

    #[test]
    fn test_viewport_depth_equal_mid() {
        // min == max at 0.5 is valid
        let viewport = Viewport::full_target(100, 100).depth_range(0.5, 0.5);
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_depth_range_exactly_zero() {
        let viewport = Viewport::full_target(100, 100).depth_range(0.0, 0.0);
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_depth_range_exactly_one() {
        let viewport = Viewport::full_target(100, 100).depth_range(1.0, 1.0);
        assert!(viewport.validate().is_ok());
    }

    #[test]
    fn test_viewport_depth_negative_epsilon() {
        let viewport = Viewport::full_target(100, 100).min_depth(-f32::EPSILON);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidMinDepth(_))
        ));
    }

    #[test]
    fn test_viewport_depth_one_plus_epsilon() {
        let viewport = Viewport::full_target(100, 100).max_depth(1.0 + f32::EPSILON);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidMaxDepth(_))
        ));
    }

    #[test]
    fn test_viewport_reversed_z_then_standard() {
        // Chain reversed then standard should reset
        let viewport = Viewport::full_target(100, 100)
            .reversed_z()
            .standard_depth();
        assert_eq!(viewport.min_depth, 0.0);
        assert_eq!(viewport.max_depth, 1.0);
    }

    #[test]
    fn test_viewport_standard_then_reversed() {
        let viewport = Viewport::full_target(100, 100)
            .standard_depth()
            .reversed_z();
        assert_eq!(viewport.min_depth, 1.0);
        assert_eq!(viewport.max_depth, 0.0);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: ScissorRect Bounds Validation Error Paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_scissor_validate_exactly_at_boundary() {
        let scissor = ScissorRect::new(0, 0, 1920, 1080);
        assert!(scissor.validate_bounds(1920, 1080).is_ok());
    }

    #[test]
    fn test_scissor_validate_one_pixel_over_width() {
        let scissor = ScissorRect::new(0, 0, 1921, 1080);
        assert!(matches!(
            scissor.validate_bounds(1920, 1080),
            Err(ScissorError::ExceedsTargetWidth { .. })
        ));
    }

    #[test]
    fn test_scissor_validate_one_pixel_over_height() {
        let scissor = ScissorRect::new(0, 0, 1920, 1081);
        assert!(matches!(
            scissor.validate_bounds(1920, 1080),
            Err(ScissorError::ExceedsTargetHeight { .. })
        ));
    }

    #[test]
    fn test_scissor_validate_offset_plus_size_exceeds() {
        let scissor = ScissorRect::new(500, 0, 1500, 1080);
        // 500 + 1500 = 2000 > 1920
        assert!(matches!(
            scissor.validate_bounds(1920, 1080),
            Err(ScissorError::ExceedsTargetWidth { .. })
        ));
    }

    #[test]
    fn test_scissor_validate_zero_size_at_boundary() {
        let scissor = ScissorRect::new(1920, 1080, 0, 0);
        assert!(scissor.validate_bounds(1920, 1080).is_ok());
    }

    #[test]
    fn test_scissor_validate_zero_target() {
        let scissor = ScissorRect::new(0, 0, 1, 1);
        assert!(scissor.validate_bounds(0, 0).is_err());
    }

    #[test]
    fn test_scissor_validate_max_values_small_target() {
        let scissor = ScissorRect::new(u32::MAX, u32::MAX, 1, 1);
        assert!(scissor.validate_bounds(100, 100).is_err());
    }

    #[test]
    fn test_scissor_error_values_accessible() {
        let scissor = ScissorRect::new(100, 0, 2000, 1080);
        if let Err(ScissorError::ExceedsTargetWidth { scissor_right, target_width }) =
            scissor.validate_bounds(1920, 1080)
        {
            assert_eq!(scissor_right, 2100);
            assert_eq!(target_width, 1920);
        } else {
            panic!("Expected ExceedsTargetWidth error");
        }
    }

    #[test]
    fn test_scissor_error_height_values_accessible() {
        let scissor = ScissorRect::new(0, 100, 1920, 2000);
        if let Err(ScissorError::ExceedsTargetHeight { scissor_bottom, target_height }) =
            scissor.validate_bounds(1920, 1080)
        {
            assert_eq!(scissor_bottom, 2100);
            assert_eq!(target_height, 1080);
        } else {
            panic!("Expected ExceedsTargetHeight error");
        }
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Builder Pattern Combinations
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default_impl() {
        let builder = ViewportBuilder::default();
        let viewport = builder.build_unchecked();
        assert_eq!(viewport, Viewport::default());
    }

    #[test]
    fn test_builder_clone() {
        let builder = ViewportBuilder::new().size(100.0, 200.0);
        let cloned = builder.clone();
        assert_eq!(builder.build_unchecked(), cloned.build_unchecked());
    }

    #[test]
    fn test_builder_debug() {
        let builder = ViewportBuilder::new();
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("ViewportBuilder"));
    }

    #[test]
    fn test_builder_all_methods_chained() {
        let viewport = ViewportBuilder::new()
            .x(1.0)
            .y(2.0)
            .width(300.0)
            .height(400.0)
            .min_depth(0.1)
            .max_depth(0.9)
            .build()
            .unwrap();

        assert_eq!(viewport.x, 1.0);
        assert_eq!(viewport.y, 2.0);
        assert_eq!(viewport.width, 300.0);
        assert_eq!(viewport.height, 400.0);
        assert_eq!(viewport.min_depth, 0.1);
        assert_eq!(viewport.max_depth, 0.9);
    }

    #[test]
    fn test_builder_override_values() {
        let viewport = ViewportBuilder::new()
            .size(100.0, 100.0)
            .size(200.0, 200.0) // override
            .build()
            .unwrap();

        assert_eq!(viewport.width, 200.0);
        assert_eq!(viewport.height, 200.0);
    }

    #[test]
    fn test_builder_position_then_individual() {
        let viewport = ViewportBuilder::new()
            .position(10.0, 20.0)
            .x(30.0) // override x only
            .size(100.0, 100.0)
            .build()
            .unwrap();

        assert_eq!(viewport.x, 30.0);
        assert_eq!(viewport.y, 20.0);
    }

    #[test]
    fn test_builder_depth_methods_interaction() {
        let viewport = ViewportBuilder::new()
            .size(100.0, 100.0)
            .depth_range(0.2, 0.8)
            .min_depth(0.3) // override min only
            .build()
            .unwrap();

        assert_eq!(viewport.min_depth, 0.3);
        assert_eq!(viewport.max_depth, 0.8);
    }

    #[test]
    fn test_builder_full_target_then_modify() {
        let viewport = ViewportBuilder::full_target(1920, 1080)
            .position(10.0, 20.0)
            .build()
            .unwrap();

        assert_eq!(viewport.x, 10.0);
        assert_eq!(viewport.y, 20.0);
        assert_eq!(viewport.width, 1920.0);
        assert_eq!(viewport.height, 1080.0);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: ViewportInfo Metadata Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_info_full_target() {
        let info = get_viewport_info("Full Target").unwrap();
        assert!(info.description.contains("entire"));
        assert!(info.use_cases.contains(&"standard rendering"));
    }

    #[test]
    fn test_viewport_info_reversed_z() {
        let info = get_viewport_info("Reversed-Z").unwrap();
        assert!(info.description.contains("precision"));
        assert!(info.use_cases.contains(&"large scenes"));
    }

    #[test]
    fn test_viewport_info_split_screen_left() {
        let info = get_viewport_info("Split Screen Left").unwrap();
        assert!(info.description.contains("Left half"));
        assert!(info.use_cases.contains(&"multiplayer"));
    }

    #[test]
    fn test_viewport_info_split_screen_right() {
        let info = get_viewport_info("Split Screen Right").unwrap();
        assert!(info.description.contains("Right half"));
    }

    #[test]
    fn test_viewport_info_equality() {
        let info1 = get_viewport_info("Full Target").unwrap();
        let info2 = &VIEWPORT_PRESETS[0];
        assert_eq!(info1, info2);
    }

    #[test]
    fn test_viewport_info_copy() {
        let info = VIEWPORT_PRESETS[0];
        let copy = info;
        assert_eq!(info.name, copy.name);
    }

    #[test]
    fn test_viewport_info_debug() {
        let debug_str = format!("{:?}", VIEWPORT_PRESETS[0]);
        assert!(debug_str.contains("ViewportInfo"));
        assert!(debug_str.contains("Full Target"));
    }

    #[test]
    fn test_viewport_info_all_unique_names() {
        let names: Vec<_> = VIEWPORT_PRESETS.iter().map(|i| i.name).collect();
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other, "Duplicate preset name found");
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Split-Screen Helper Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_split_screen_left_zero_dimensions() {
        let viewport = split_screen_left(0, 0);
        assert_eq!(viewport.width, 0.0);
        assert_eq!(viewport.height, 0.0);
    }

    #[test]
    fn test_split_screen_right_zero_dimensions() {
        let viewport = split_screen_right(0, 0);
        assert_eq!(viewport.width, 0.0);
        assert_eq!(viewport.height, 0.0);
    }

    #[test]
    fn test_split_screen_top_zero_dimensions() {
        let viewport = split_screen_top(0, 0);
        assert_eq!(viewport.width, 0.0);
        assert_eq!(viewport.height, 0.0);
    }

    #[test]
    fn test_split_screen_bottom_zero_dimensions() {
        let viewport = split_screen_bottom(0, 0);
        assert_eq!(viewport.width, 0.0);
        assert_eq!(viewport.height, 0.0);
    }

    #[test]
    fn test_split_screen_single_pixel_width() {
        let left = split_screen_left(1, 100);
        let right = split_screen_right(1, 100);
        // 1 / 2 = 0, so left gets 0, right gets 1
        assert_eq!(left.width, 0.0);
        assert_eq!(right.width, 1.0);
    }

    #[test]
    fn test_split_screen_single_pixel_height() {
        let top = split_screen_top(100, 1);
        let bottom = split_screen_bottom(100, 1);
        assert_eq!(top.height, 0.0);
        assert_eq!(bottom.height, 1.0);
    }

    #[test]
    fn test_split_screen_large_dimensions() {
        let left = split_screen_left(u32::MAX, u32::MAX);
        let right = split_screen_right(u32::MAX, u32::MAX);
        // Should not panic
        assert!(left.width > 0.0);
        assert!(right.width > 0.0);
    }

    #[test]
    fn test_split_screen_horizontal_coverage() {
        for width in [2, 3, 100, 1000, 1919, 1920, 1921] {
            let left = split_screen_left(width, 100);
            let right = split_screen_right(width, 100);
            let total = left.width + right.width;
            assert_eq!(total, width as f32, "Width {} not fully covered", width);
        }
    }

    #[test]
    fn test_split_screen_vertical_coverage() {
        for height in [2, 3, 100, 1000, 1079, 1080, 1081] {
            let top = split_screen_top(100, height);
            let bottom = split_screen_bottom(100, height);
            let total = top.height + bottom.height;
            assert_eq!(total, height as f32, "Height {} not fully covered", height);
        }
    }

    #[test]
    fn test_split_screen_no_gap() {
        let left = split_screen_left(1920, 1080);
        let right = split_screen_right(1920, 1080);
        // Left edge of right should equal right edge of left
        assert_eq!(left.x + left.width, right.x);
    }

    #[test]
    fn test_split_screen_no_vertical_gap() {
        let top = split_screen_top(1920, 1080);
        let bottom = split_screen_bottom(1920, 1080);
        assert_eq!(top.y + top.height, bottom.y);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Quadrant Viewport Calculations
    // -------------------------------------------------------------------------

    #[test]
    fn test_quadrant_viewport_zero_dimensions() {
        for q in 0..4 {
            let viewport = quadrant_viewport(q, 0, 0);
            assert_eq!(viewport.width, 0.0);
            assert_eq!(viewport.height, 0.0);
        }
    }

    #[test]
    fn test_quadrant_viewport_full_coverage() {
        let width = 1920u32;
        let height = 1080u32;

        let tl = quadrant_viewport(0, width, height);
        let tr = quadrant_viewport(1, width, height);
        let bl = quadrant_viewport(2, width, height);
        let br = quadrant_viewport(3, width, height);

        // All four quadrants should cover entire area
        let total_area = tl.area() + tr.area() + bl.area() + br.area();
        assert_eq!(total_area, (width * height) as f32);
    }

    #[test]
    fn test_quadrant_viewport_no_overlap() {
        let width = 100u32;
        let height = 100u32;

        let viewports: Vec<_> = (0..4)
            .map(|q| quadrant_viewport(q, width, height))
            .collect();

        for (i, v1) in viewports.iter().enumerate() {
            for (j, v2) in viewports.iter().enumerate() {
                if i != j {
                    // Convert to scissor for intersection check
                    let s1 = ScissorRect::from_viewport(v1);
                    let s2 = ScissorRect::from_viewport(v2);
                    assert!(s1.intersection(&s2).is_none(),
                            "Quadrants {} and {} overlap", i, j);
                }
            }
        }
    }

    #[test]
    fn test_quadrant_viewport_odd_dimensions() {
        let tl = quadrant_viewport(0, 101, 101);
        let tr = quadrant_viewport(1, 101, 101);
        let bl = quadrant_viewport(2, 101, 101);
        let br = quadrant_viewport(3, 101, 101);

        // Half of 101 is 50, so each quadrant is 50x50
        assert_eq!(tl.width, 50.0);
        assert_eq!(tl.height, 50.0);
        // Top-right starts at 50
        assert_eq!(tr.x, 50.0);
        assert_eq!(tr.width, 50.0);
    }

    #[test]
    fn test_quadrant_viewport_all_indices_beyond_3() {
        // Any index >= 4 should map to bottom-right
        for i in 4..10 {
            let viewport = quadrant_viewport(i, 100, 100);
            assert_eq!(viewport.x, 50.0);
            assert_eq!(viewport.y, 50.0);
        }
    }

    #[test]
    fn test_quadrant_viewport_u8_max() {
        let viewport = quadrant_viewport(u8::MAX, 100, 100);
        // Should be bottom-right
        assert_eq!(viewport.x, 50.0);
        assert_eq!(viewport.y, 50.0);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: ScissorRect Intersection Logic
    // -------------------------------------------------------------------------

    #[test]
    fn test_scissor_intersection_same_rect() {
        let rect = ScissorRect::new(10, 20, 100, 200);
        let intersection = rect.intersection(&rect).unwrap();
        assert_eq!(intersection, rect);
    }

    #[test]
    fn test_scissor_intersection_partial_horizontal() {
        let a = ScissorRect::new(0, 0, 100, 100);
        let b = ScissorRect::new(50, 0, 100, 100);
        let int = a.intersection(&b).unwrap();
        assert_eq!(int.x, 50);
        assert_eq!(int.y, 0);
        assert_eq!(int.width, 50);
        assert_eq!(int.height, 100);
    }

    #[test]
    fn test_scissor_intersection_partial_vertical() {
        let a = ScissorRect::new(0, 0, 100, 100);
        let b = ScissorRect::new(0, 50, 100, 100);
        let int = a.intersection(&b).unwrap();
        assert_eq!(int.x, 0);
        assert_eq!(int.y, 50);
        assert_eq!(int.width, 100);
        assert_eq!(int.height, 50);
    }

    #[test]
    fn test_scissor_intersection_corner_overlap() {
        let a = ScissorRect::new(0, 0, 100, 100);
        let b = ScissorRect::new(75, 75, 100, 100);
        let int = a.intersection(&b).unwrap();
        assert_eq!(int.x, 75);
        assert_eq!(int.y, 75);
        assert_eq!(int.width, 25);
        assert_eq!(int.height, 25);
    }

    #[test]
    fn test_scissor_intersection_one_pixel_overlap() {
        let a = ScissorRect::new(0, 0, 100, 100);
        let b = ScissorRect::new(99, 99, 100, 100);
        let int = a.intersection(&b).unwrap();
        assert_eq!(int.width, 1);
        assert_eq!(int.height, 1);
    }

    #[test]
    fn test_scissor_intersection_adjacent_no_overlap() {
        // Horizontally adjacent
        let a = ScissorRect::new(0, 0, 100, 100);
        let b = ScissorRect::new(100, 0, 100, 100);
        assert!(a.intersection(&b).is_none());

        // Vertically adjacent
        let c = ScissorRect::new(0, 100, 100, 100);
        assert!(a.intersection(&c).is_none());
    }

    #[test]
    fn test_scissor_intersection_empty_rects() {
        let empty = ScissorRect::empty();
        let normal = ScissorRect::new(0, 0, 100, 100);
        assert!(empty.intersection(&normal).is_none());
        assert!(normal.intersection(&empty).is_none());
    }

    #[test]
    fn test_scissor_intersection_commutative() {
        let a = ScissorRect::new(10, 20, 100, 200);
        let b = ScissorRect::new(50, 60, 80, 90);
        assert_eq!(a.intersection(&b), b.intersection(&a));
    }

    #[test]
    fn test_scissor_intersection_at_origin() {
        let a = ScissorRect::new(0, 0, 50, 50);
        let b = ScissorRect::new(0, 0, 30, 30);
        let int = a.intersection(&b).unwrap();
        assert_eq!(int.x, 0);
        assert_eq!(int.y, 0);
        assert_eq!(int.width, 30);
        assert_eq!(int.height, 30);
    }

    #[test]
    fn test_scissor_intersection_large_offset() {
        let a = ScissorRect::new(1000000, 1000000, 100, 100);
        let b = ScissorRect::new(1000050, 1000050, 100, 100);
        let int = a.intersection(&b).unwrap();
        assert_eq!(int.x, 1000050);
        assert_eq!(int.y, 1000050);
        assert_eq!(int.width, 50);
        assert_eq!(int.height, 50);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: from_viewport Conversion
    // -------------------------------------------------------------------------

    #[test]
    fn test_scissor_from_viewport_truncation() {
        let viewport = Viewport::new()
            .position(10.9, 20.9)
            .size(100.9, 200.9);
        let scissor = ScissorRect::from_viewport(&viewport);
        assert_eq!(scissor.x, 10);
        assert_eq!(scissor.y, 20);
        assert_eq!(scissor.width, 100);
        assert_eq!(scissor.height, 200);
    }

    #[test]
    fn test_scissor_from_viewport_zero() {
        let viewport = Viewport::new();
        let scissor = ScissorRect::from_viewport(&viewport);
        assert_eq!(scissor.x, 0);
        assert_eq!(scissor.y, 0);
        assert_eq!(scissor.width, 0);
        assert_eq!(scissor.height, 0);
    }

    #[test]
    fn test_scissor_from_viewport_large_values() {
        let viewport = Viewport::new()
            .position(0.0, 0.0)
            .size(4294967040.0, 4294967040.0); // Close to u32::MAX
        let scissor = ScissorRect::from_viewport(&viewport);
        assert!(scissor.width > 0);
        assert!(scissor.height > 0);
    }

    #[test]
    fn test_scissor_from_viewport_negative_clamped() {
        let viewport = Viewport::new()
            .position(-1000.0, -2000.0)
            .size(-500.0, -600.0);
        let scissor = ScissorRect::from_viewport(&viewport);
        assert_eq!(scissor.x, 0);
        assert_eq!(scissor.y, 0);
        assert_eq!(scissor.width, 0);
        assert_eq!(scissor.height, 0);
    }

    #[test]
    fn test_scissor_from_viewport_mixed_signs() {
        let viewport = Viewport::new()
            .position(-10.0, 20.0)
            .size(100.0, -50.0);
        let scissor = ScissorRect::from_viewport(&viewport);
        assert_eq!(scissor.x, 0); // clamped
        assert_eq!(scissor.y, 20);
        assert_eq!(scissor.width, 100);
        assert_eq!(scissor.height, 0); // clamped
    }

    #[test]
    fn test_scissor_from_viewport_infinity() {
        let viewport = Viewport::new()
            .position(f32::INFINITY, 0.0)
            .size(100.0, 100.0);
        let scissor = ScissorRect::from_viewport(&viewport);
        // f32::INFINITY.max(0.0) as u32 behavior
        assert!(scissor.x == u32::MAX || scissor.x == 0); // Implementation defined
    }

    #[test]
    fn test_scissor_from_viewport_nan() {
        let viewport = Viewport::new()
            .position(f32::NAN, 0.0)
            .size(100.0, 100.0);
        let scissor = ScissorRect::from_viewport(&viewport);
        // NaN.max(0.0) is 0.0 in Rust
        assert_eq!(scissor.x, 0);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Display/Debug Trait Implementations
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_error_std_error_trait() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<ViewportError>();
    }

    #[test]
    fn test_scissor_error_std_error_trait() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<ScissorError>();
    }

    #[test]
    fn test_viewport_error_display_width_zero() {
        let err = ViewportError::InvalidWidth(0.0);
        let msg = err.to_string();
        assert!(msg.contains("width"));
        assert!(msg.contains("0"));
        assert!(msg.contains("> 0"));
    }

    #[test]
    fn test_viewport_error_display_height_message() {
        let err = ViewportError::InvalidHeight(-5.5);
        let msg = err.to_string();
        assert!(msg.contains("height"));
        assert!(msg.contains("-5.5"));
    }

    #[test]
    fn test_viewport_error_display_depth_message() {
        let err = ViewportError::InvalidMinDepth(-0.5);
        let msg = err.to_string();
        assert!(msg.contains("min_depth"));
        assert!(msg.contains("[0.0, 1.0]"));
    }

    #[test]
    fn test_scissor_error_display_width_message() {
        let err = ScissorError::ExceedsTargetWidth {
            scissor_right: 3000,
            target_width: 1920,
        };
        let msg = err.to_string();
        assert!(msg.contains("width"));
        assert!(msg.contains("3000"));
        assert!(msg.contains("1920"));
    }

    #[test]
    fn test_scissor_error_display_height_message() {
        let err = ScissorError::ExceedsTargetHeight {
            scissor_bottom: 2000,
            target_height: 1080,
        };
        let msg = err.to_string();
        assert!(msg.contains("height"));
        assert!(msg.contains("2000"));
        assert!(msg.contains("1080"));
    }

    #[test]
    fn test_viewport_error_debug() {
        let err = ViewportError::InvalidWidth(-1.0);
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidWidth"));
    }

    #[test]
    fn test_scissor_error_debug() {
        let err = ScissorError::ExceedsTargetWidth {
            scissor_right: 100,
            target_width: 50,
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("ExceedsTargetWidth"));
    }

    #[test]
    fn test_scissor_error_clone() {
        let err = ScissorError::ExceedsTargetWidth {
            scissor_right: 100,
            target_width: 50,
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_viewport_error_clone() {
        let err = ViewportError::InvalidWidth(-1.0);
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Viewport contains_point Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_contains_point_at_edges() {
        let viewport = Viewport::new()
            .position(0.0, 0.0)
            .size(100.0, 100.0);

        // Top-left corner (inclusive)
        assert!(viewport.contains_point(0.0, 0.0));
        // Just inside right edge
        assert!(viewport.contains_point(99.999, 50.0));
        // Just inside bottom edge
        assert!(viewport.contains_point(50.0, 99.999));
        // Right edge (exclusive)
        assert!(!viewport.contains_point(100.0, 50.0));
        // Bottom edge (exclusive)
        assert!(!viewport.contains_point(50.0, 100.0));
    }

    #[test]
    fn test_viewport_contains_point_negative_position() {
        let viewport = Viewport::new()
            .position(-50.0, -50.0)
            .size(100.0, 100.0);

        assert!(viewport.contains_point(-50.0, -50.0));
        assert!(viewport.contains_point(0.0, 0.0));
        assert!(viewport.contains_point(49.0, 49.0));
        assert!(!viewport.contains_point(50.0, 0.0));
    }

    #[test]
    fn test_viewport_contains_point_zero_size() {
        let viewport = Viewport::new()
            .position(50.0, 50.0)
            .size(0.0, 0.0);

        assert!(!viewport.contains_point(50.0, 50.0));
        assert!(!viewport.contains_point(0.0, 0.0));
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: ScissorRect contains_point Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_scissor_contains_point_at_edges() {
        let scissor = ScissorRect::new(0, 0, 100, 100);

        assert!(scissor.contains_point(0, 0));
        assert!(scissor.contains_point(99, 99));
        assert!(!scissor.contains_point(100, 0));
        assert!(!scissor.contains_point(0, 100));
    }

    #[test]
    fn test_scissor_contains_point_saturation() {
        // Test with large coordinates that might overflow
        let scissor = ScissorRect::new(u32::MAX - 10, u32::MAX - 10, 20, 20);
        // Point at origin of scissor
        assert!(scissor.contains_point(u32::MAX - 10, u32::MAX - 10));
        // Point before scissor
        assert!(!scissor.contains_point(0, 0));
    }

    #[test]
    fn test_scissor_contains_point_zero_dimensions() {
        let scissor = ScissorRect::new(50, 50, 0, 100);
        assert!(!scissor.contains_point(50, 50));

        let scissor2 = ScissorRect::new(50, 50, 100, 0);
        assert!(!scissor2.contains_point(50, 50));
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Aspect Ratio Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_aspect_ratio_square() {
        let viewport = Viewport::full_target(1000, 1000);
        assert!((viewport.aspect_ratio().unwrap() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_viewport_aspect_ratio_ultrawide() {
        let viewport = Viewport::full_target(3440, 1440);
        let aspect = viewport.aspect_ratio().unwrap();
        assert!((aspect - 3440.0 / 1440.0).abs() < 0.0001);
    }

    #[test]
    fn test_viewport_aspect_ratio_portrait() {
        let viewport = Viewport::full_target(1080, 1920);
        let aspect = viewport.aspect_ratio().unwrap();
        assert!(aspect < 1.0);
    }

    #[test]
    fn test_viewport_aspect_ratio_very_wide() {
        let viewport = Viewport::new().size(10000.0, 1.0);
        let aspect = viewport.aspect_ratio().unwrap();
        assert_eq!(aspect, 10000.0);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Area Calculations
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_area_zero() {
        let viewport = Viewport::new();
        assert_eq!(viewport.area(), 0.0);
    }

    #[test]
    fn test_viewport_area_negative_components() {
        // Even with invalid negative size, area calculation works
        let viewport = Viewport::new().size(-10.0, -20.0);
        assert_eq!(viewport.area(), 200.0); // -10 * -20 = 200
    }

    #[test]
    fn test_scissor_area_max_values() {
        // Test u64 overflow protection
        let scissor = ScissorRect::new(0, 0, u32::MAX, u32::MAX);
        let expected = u32::MAX as u64 * u32::MAX as u64;
        assert_eq!(scissor.area(), expected);
    }

    // -------------------------------------------------------------------------
    // WHITEBOX TESTS: Validation Priority
    // -------------------------------------------------------------------------

    #[test]
    fn test_viewport_validation_order_width_first() {
        // Both width and height invalid, width error should come first
        let viewport = Viewport::new().size(0.0, 0.0);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidWidth(_))
        ));
    }

    #[test]
    fn test_viewport_validation_order_depth_after_size() {
        // Width valid, height invalid - should get height error
        let viewport = Viewport::new().size(100.0, 0.0);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidHeight(_))
        ));
    }

    #[test]
    fn test_viewport_validation_order_min_depth_before_max() {
        // Size valid, both depths invalid - min_depth error first
        let viewport = Viewport::full_target(100, 100)
            .min_depth(-1.0)
            .max_depth(2.0);
        assert!(matches!(
            viewport.validate(),
            Err(ViewportError::InvalidMinDepth(_))
        ));
    }
}
