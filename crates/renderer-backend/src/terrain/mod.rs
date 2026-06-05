//! Terrain Rendering Subsystem
//!
//! This module provides GPU-side structures for terrain rendering:
//!
//! - [`Clipmap`] — LOD management using geoclipmaps
//! - [`ClipmapConfig`] — GPU-compatible configuration structure
//! - [`ClipmapLevel`] — Per-level spatial tracking
//! - [`ClipmapUpdateRegion`] — Incremental update regions
//! - [`Geomorphing`] — Smooth LOD transition utilities
//! - [`ClipmapVertex`] — GPU vertex format with morph factor
//! - [`SplatMap`] — Material blending with up to 8 layers
//! - [`TerrainLayerDef`] — Per-layer height/slope auto-blending
//! - [`TerrainMaterialConfig`] — GPU-compatible material configuration
//!
//! # Clipmap Overview
//!
//! Clipmaps provide efficient LOD for large terrains by rendering concentric
//! rings of geometry centered on the camera. Each ring has 2x the spacing of
//! the ring inside it, providing O(log N) complexity for terrain size N.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::terrain::{Clipmap, ClipmapConfig};
//!
//! let config = ClipmapConfig::new(128, 8, 0.5, 100.0, 500.0);
//! let mut clipmap = Clipmap::new(config);
//!
//! // Each frame:
//! clipmap.update_camera(camera_position);
//!
//! for region in clipmap.compute_update_regions() {
//!     // Fetch heightfield data for region
//!     // Upload to GPU ring buffer
//! }
//! ```

pub mod clipmap;
pub mod terrain_material;

pub use clipmap::{
    // Configuration
    ClipmapConfig,
    ClipmapError,

    // Core types
    Clipmap,
    ClipmapLevel,
    ClipmapUpdateRegion,

    // GPU vertex format
    ClipmapVertex,

    // Utilities
    Geomorphing,
    ToroidalAddress,

    // Normal computation
    compute_normal_central_diff,
    compute_normal_forward_diff,
    compute_tangent_frame,

    // Constants
    DEFAULT_GRID_SIZE,
    DEFAULT_NUM_LEVELS,
    DEFAULT_FINEST_SPACING,
    DEFAULT_HEIGHT_SCALE,
    DEFAULT_MAX_HEIGHT,
    MIN_GRID_SIZE,
    MAX_GRID_SIZE,
    MIN_LEVELS,
    MAX_LEVELS,
    UPDATE_THRESHOLD_RATIO,
};

pub use terrain_material::{
    // Configuration
    TerrainMaterialConfig,
    TerrainMaterialError,
    TerrainLayerDef,

    // Core types
    SplatMap,
    SplatPixel,
    StochasticParams,
    HexCell,

    // Utility functions
    smoothstep,
    smootherstep,
    slope_from_normal,
    slope_from_gradient,
    compute_normal_from_heights,

    // Constants
    MAX_TERRAIN_LAYERS,
    DEFAULT_SPLAT_RESOLUTION,
    DEFAULT_UV_SCALE,
    MIN_SPLAT_RESOLUTION,
    MAX_SPLAT_RESOLUTION,
    MIN_UV_SCALE,
    MAX_UV_SCALE,
    WEIGHT_EPSILON,
    DEFAULT_BLEND_FALLOFF,
    GOLDEN_RATIO,
    STOCHASTIC_SCALE,
};
