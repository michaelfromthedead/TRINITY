//! Omega rendering runtime.
//!
//! This crate provides:
//! - Deterministic fixed-point math (`fixed`, `vec`, `mat`, `quat`)
//! - Trigonometry LUT (`trig`) and PRNG (`rng`) for cross-platform reproducibility
//! - Spatial primitives (`spatial`) for collision and culling
//! - RHI abstraction layer (`rhi`) for GPU backends
//!
//! The `bridge` module requires the `pyo3` feature and is not built by default.

pub mod fixed;
pub mod vec;
pub mod mat;
pub mod quat;
pub mod trig;
pub mod rng;
pub mod spatial;
pub mod rhi;

// bridge module requires pyo3 + renderer_backend - not built by default
// Enable with: cargo build --features pyo3
#[cfg(feature = "pyo3")]
pub mod bridge;

// Re-export core types at crate root for convenience
pub use fixed::{Fixed16, Fixed32};
pub use vec::{Vec2, Vec3, Vec4, FVec2, FVec3, FVec4};
pub use mat::{Mat3, Mat4, M64};
pub use quat::{Quat, FQuat};
pub use trig::TrigLUT;
pub use rng::SimRng;
pub use spatial::{AABB, Frustum, Ray};
