//! Demoscene WGSL shaders for SDF and noise operations.
//!
//! These shaders are embedded as strings for compile-time validation.

pub const NOISE_HASH: &str = include_str!("noise_hash.wgsl");
pub const NOISE_VALUE: &str = include_str!("noise_value.wgsl");
pub const NOISE_PERLIN: &str = include_str!("noise_perlin.wgsl");
pub const NOISE_FBM: &str = include_str!("noise_fbm.wgsl");
pub const NOISE_RIDGED: &str = include_str!("noise_ridged.wgsl");
pub const NOISE_DOMAIN_WARP: &str = include_str!("noise_domain_warp.wgsl");
pub const SDF_DOMAIN: &str = include_str!("sdf_domain.wgsl");
