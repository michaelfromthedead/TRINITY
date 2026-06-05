//! Global Illumination (GI) subsystem for DDGI probe volumes.
//!
//! This module provides GPU-side structures for probe-based global illumination:
//!
//! - [`ProbeGridGpu`] — GPU metadata for probe volume grid layout
//! - [`ProbeSH`] — Per-probe spherical harmonics irradiance storage
//! - [`ProbeVis`] — Per-probe visibility/occlusion data
//! - [`ProbeRingBuffer`] — Ring buffer for infinite scrolling volumes
//! - [`DDGIQuality`] — Quality presets with fixed-size allocations
//! - [`DDGIAllocation`] — Pre-allocated GPU buffers for probe volumes
//! - [`DDGISamplePass`] — Compute pass for sampling probe irradiance
//! - [`DDGIScrollManager`] — Infinite scrolling volume manager (T-GIR-P2.6)
//! - [`DDGIRasterizedPass`] — Rasterized fallback when hardware RT unavailable (T-GIR-P2.3)

pub mod ddgi_allocator;
pub mod ddgi_rasterized;
pub mod ddgi_sample;
pub mod ddgi_scroll;
pub mod probe_grid;

pub use ddgi_allocator::{
    AllocationResult,
    BufferHandle,
    DDGIAllocation,
    DDGIAllocator,
    DDGIQuality,
    MockDDGIAllocator,
    create_irradiance_buffer,
    create_probe_grid_gpu,
    create_visibility_buffer,
};
pub use ddgi_sample::{
    DDGICameraUniforms,
    DDGIDebugMode,
    DDGISampleBindGroupLayoutDesc,
    DDGISampleBindings,
    DDGISampleConfig,
    DDGISamplePass,
    DDGISampleQuality,
    DDGISampleResources,
    apply_scroll_offset,
    compute_trilinear_weights,
    create_ddgi_sample_pass,
    verify_trilinear_weights,
};
pub use ddgi_rasterized::{
    AtlasFormat,
    AtlasTextureDesc,
    AtlasUsage,
    AtlasUV,
    CubemapFace,
    DDGIRasterizedConfig,
    DDGIRasterizedPass,
    ProbeBatchScheduler,
    cubemap_projection,
    direction_from_face_uv,
    get_face_view_matrices,
    get_face_view_matrices_flat,
    get_face_view_matrix,
    look_at,
    sh_basis_l2,
    sh_project_sample,
    texel_solid_angle,
};
pub use ddgi_scroll::{
    DDGIScrollManager,
    GridShiftParams,
    ProbeStatus,
    ScrollDelta,
    ScrollHysteresis,
    seed_probe_from_neighbors_cpu,
};
pub use probe_grid::{
    ProbeGridGpu,
    ProbeSH,
    ProbeVis,
    ProbeRingBuffer,
};
