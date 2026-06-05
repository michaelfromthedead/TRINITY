//! GPU-compatible light type definitions for the TRINITY renderer.
//!
//! This module defines all light types matching the Python implementation in
//! `engine/rendering/lighting/light_types.py`. All structs use `#[repr(C)]`
//! for GPU buffer compatibility and predictable memory layout.
//!
//! Light types supported:
//! - DirectionalLight (sun/moon with CSM shadows)
//! - PointLight (position, radius, cube shadows)
//! - SpotLight (position, direction, inner/outer angle)
//! - RectAreaLight (LTC-based rectangular area light)
//! - DiskAreaLight (LTC-based disk area light)
//! - IESLight (IES profile data)
//! - SkyLight (cubemap-based ambient)

use std::f32::consts::PI;

/// Light type enumeration matching Python `LightType` enum.
/// Uses `#[repr(u32)]` for GPU shader compatibility.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LightTypeGPU {
    Directional = 0,
    Point = 1,
    Spot = 2,
    RectArea = 3,
    DiskArea = 4,
    IES = 5,
    Sky = 6,
}

impl LightTypeGPU {
    /// Convert from u32 value (for GPU readback).
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Directional),
            1 => Some(Self::Point),
            2 => Some(Self::Spot),
            3 => Some(Self::RectArea),
            4 => Some(Self::DiskArea),
            5 => Some(Self::IES),
            6 => Some(Self::Sky),
            _ => None,
        }
    }
}

/// Shadow mode enumeration matching Python `ShadowMode`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShadowModeGPU {
    #[default]
    None = 0,
    Static = 1,
    Dynamic = 2,
}

/// GI importance level matching Python `GIImportance`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GIImportanceGPU {
    Low = 0,
    #[default]
    Medium = 1,
    High = 2,
    Critical = 3,
}

// =============================================================================
// Light Struct Definitions
// =============================================================================

/// Directional light (sun/moon) with cascaded shadow maps.
///
/// Size: 64 bytes (16-byte aligned for GPU)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectionalLightGPU {
    /// Normalized direction the light is shining (vec3 + padding)
    pub direction: [f32; 3],
    /// Angular diameter in radians (for soft shadows), default ~0.00935 (sun)
    pub angular_diameter: f32,

    /// Light color RGB in [0,1] range
    pub color: [f32; 3],
    /// Light intensity multiplier
    pub intensity: f32,

    /// Number of CSM cascades (1-4)
    pub cascade_count: u32,
    /// Shadow mode (None/Static/Dynamic)
    pub shadow_mode: u32,
    /// Shadow bias for depth comparison
    pub shadow_bias: f32,
    /// Resolution scale for shadow maps
    pub shadow_resolution_scale: f32,

    /// Cascade split distances (max 4 cascades)
    pub cascade_distances: [f32; 4],
}

// Static size assertion: 64 bytes
const _: () = assert!(std::mem::size_of::<DirectionalLightGPU>() == 64);
const _: () = assert!(std::mem::align_of::<DirectionalLightGPU>() == 4);

/// Point light with cube shadow mapping.
///
/// Size: 48 bytes (16-byte aligned for GPU)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PointLightGPU {
    /// World position of the light
    pub position: [f32; 3],
    /// Light influence radius (attenuation reaches 0 at radius)
    pub radius: f32,

    /// Light color RGB in [0,1] range
    pub color: [f32; 3],
    /// Light intensity multiplier (lumens)
    pub intensity: f32,

    /// Attenuation falloff exponent (default 2.0 for inverse-square)
    pub falloff_exponent: f32,
    /// Shadow mode (None/Static/Dynamic)
    pub shadow_mode: u32,
    /// Shadow bias for depth comparison
    pub shadow_bias: f32,
    /// Padding for 16-byte alignment
    pub _pad0: f32,
}

const _: () = assert!(std::mem::size_of::<PointLightGPU>() == 48);
const _: () = assert!(std::mem::align_of::<PointLightGPU>() == 4);

/// Spot light with spot shadow mapping.
///
/// Size: 64 bytes (16-byte aligned for GPU)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SpotLightGPU {
    /// World position of the light
    pub position: [f32; 3],
    /// Light influence radius
    pub radius: f32,

    /// Direction the light is pointing (normalized)
    pub direction: [f32; 3],
    /// Light intensity (candelas)
    pub intensity: f32,

    /// Light color RGB in [0,1] range
    pub color: [f32; 3],
    /// Inner cone angle in radians (full intensity)
    pub inner_angle: f32,

    /// Outer cone angle in radians (zero intensity at edge)
    pub outer_angle: f32,
    /// Attenuation falloff exponent
    pub falloff_exponent: f32,
    /// Shadow mode (None/Static/Dynamic)
    pub shadow_mode: u32,
    /// Shadow bias for depth comparison
    pub shadow_bias: f32,
}

const _: () = assert!(std::mem::size_of::<SpotLightGPU>() == 64);
const _: () = assert!(std::mem::align_of::<SpotLightGPU>() == 4);

/// Rectangular area light using Linearly Transformed Cosines (LTC).
///
/// Size: 80 bytes (16-byte aligned for GPU)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RectAreaLightGPU {
    /// Center position of the rectangle
    pub position: [f32; 3],
    /// Width of the rectangle
    pub width: f32,

    /// Normal direction of the light surface
    pub direction: [f32; 3],
    /// Height of the rectangle
    pub height: f32,

    /// Up vector defining rectangle orientation
    pub up: [f32; 3],
    /// Light intensity (nits = cd/m^2)
    pub intensity: f32,

    /// Light color RGB in [0,1] range
    pub color: [f32; 3],
    /// Whether light emits from both sides (0 or 1)
    pub two_sided: u32,

    /// GI importance level
    pub gi_importance: u32,
    /// Whether this light contributes emissive GI
    pub gi_emissive: u32,
    /// Padding for 16-byte alignment
    pub _pad0: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<RectAreaLightGPU>() == 80);
const _: () = assert!(std::mem::align_of::<RectAreaLightGPU>() == 4);

/// Disk area light using Linearly Transformed Cosines (LTC).
///
/// Size: 48 bytes (16-byte aligned for GPU)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DiskAreaLightGPU {
    /// Center position of the disk
    pub position: [f32; 3],
    /// Radius of the disk
    pub disk_radius: f32,

    /// Normal direction of the light surface
    pub direction: [f32; 3],
    /// Light intensity (nits = cd/m^2)
    pub intensity: f32,

    /// Light color RGB in [0,1] range
    pub color: [f32; 3],
    /// Whether light emits from both sides (0 or 1)
    pub two_sided: u32,
}

const _: () = assert!(std::mem::size_of::<DiskAreaLightGPU>() == 48);
const _: () = assert!(std::mem::align_of::<DiskAreaLightGPU>() == 4);

/// Light with IES profile for realistic distribution.
///
/// Size: 64 bytes (16-byte aligned for GPU)
/// Note: The actual IES profile data is stored separately in a texture.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct IESLightGPU {
    /// World position of the light
    pub position: [f32; 3],
    /// Light influence radius
    pub radius: f32,

    /// Primary direction (down in IES coordinates)
    pub direction: [f32; 3],
    /// Light intensity multiplier
    pub intensity: f32,

    /// Light color RGB in [0,1] range
    pub color: [f32; 3],
    /// Index into IES profile texture array
    pub profile_index: u32,

    /// Shadow mode (None/Static/Dynamic)
    pub shadow_mode: u32,
    /// Shadow bias for depth comparison
    pub shadow_bias: f32,
    /// Total lumens from IES profile (for fallback)
    pub profile_lumens: f32,
    /// Padding for 16-byte alignment
    pub _pad0: f32,
}

const _: () = assert!(std::mem::size_of::<IESLightGPU>() == 64);
const _: () = assert!(std::mem::align_of::<IESLightGPU>() == 4);

/// Sky light for ambient lighting from environment cubemap.
///
/// Size: 48 bytes (16-byte aligned for GPU)
/// Note: The actual cubemap is stored separately.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SkyLightGPU {
    /// Light color RGB in [0,1] range (fallback if no cubemap)
    pub color: [f32; 3],
    /// Light intensity multiplier
    pub intensity: f32,

    /// Lower hemisphere color override (if enabled)
    pub lower_hemisphere_color: [f32; 3],
    /// Whether lower hemisphere override is enabled
    pub use_lower_hemisphere: u32,

    /// Rotation angle around Y axis in radians
    pub rotation: f32,
    /// Cubemap mip levels for roughness sampling
    pub mip_count: u32,
    /// Whether to use cubemap for GI (vs just ambient)
    pub use_cubemap_for_gi: u32,
    /// Index into cubemap array
    pub cubemap_index: u32,
}

const _: () = assert!(std::mem::size_of::<SkyLightGPU>() == 48);
const _: () = assert!(std::mem::align_of::<SkyLightGPU>() == 4);

// =============================================================================
// Light Union for Polymorphic GPU Access
// =============================================================================

/// Maximum size in f32s needed to hold any light type's data.
/// Largest is RectAreaLightGPU at 80 bytes = 20 f32s.
pub const LIGHT_DATA_MAX_F32S: usize = 20;

/// Tagged union for polymorphic GPU light access.
///
/// This allows storing heterogeneous light types in a single GPU buffer
/// where shaders can dispatch based on `light_type`.
///
/// Size: 96 bytes (divisible by 16 for GPU alignment)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LightUnion {
    /// Type discriminant (LightTypeGPU as u32)
    pub light_type: u32,
    /// Light ID for indexing
    pub light_id: u32,
    /// Whether light is enabled
    pub enabled: u32,
    /// Padding for 16-byte alignment
    pub _pad: u32,
    /// Raw light data (reinterpreted based on light_type)
    pub data: [f32; LIGHT_DATA_MAX_F32S],
}

const _: () = assert!(std::mem::size_of::<LightUnion>() == 96);
const _: () = assert!(std::mem::align_of::<LightUnion>() == 4);

impl Default for LightUnion {
    fn default() -> Self {
        Self {
            light_type: LightTypeGPU::Point as u32,
            light_id: 0,
            enabled: 1,
            _pad: 0,
            data: [0.0; LIGHT_DATA_MAX_F32S],
        }
    }
}

impl LightUnion {
    /// Create a new LightUnion from a DirectionalLightGPU.
    pub fn from_directional(light: &DirectionalLightGPU, light_id: u32, enabled: bool) -> Self {
        let mut result = Self {
            light_type: LightTypeGPU::Directional as u32,
            light_id,
            enabled: enabled as u32,
            _pad: 0,
            data: [0.0; LIGHT_DATA_MAX_F32S],
        };
        // Safety: DirectionalLightGPU is repr(C) and smaller than data buffer
        unsafe {
            let src = light as *const DirectionalLightGPU as *const f32;
            let dst = result.data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(
                src,
                dst,
                std::mem::size_of::<DirectionalLightGPU>() / std::mem::size_of::<f32>(),
            );
        }
        result
    }

    /// Create a new LightUnion from a PointLightGPU.
    pub fn from_point(light: &PointLightGPU, light_id: u32, enabled: bool) -> Self {
        let mut result = Self {
            light_type: LightTypeGPU::Point as u32,
            light_id,
            enabled: enabled as u32,
            _pad: 0,
            data: [0.0; LIGHT_DATA_MAX_F32S],
        };
        unsafe {
            let src = light as *const PointLightGPU as *const f32;
            let dst = result.data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(
                src,
                dst,
                std::mem::size_of::<PointLightGPU>() / std::mem::size_of::<f32>(),
            );
        }
        result
    }

    /// Create a new LightUnion from a SpotLightGPU.
    pub fn from_spot(light: &SpotLightGPU, light_id: u32, enabled: bool) -> Self {
        let mut result = Self {
            light_type: LightTypeGPU::Spot as u32,
            light_id,
            enabled: enabled as u32,
            _pad: 0,
            data: [0.0; LIGHT_DATA_MAX_F32S],
        };
        unsafe {
            let src = light as *const SpotLightGPU as *const f32;
            let dst = result.data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(
                src,
                dst,
                std::mem::size_of::<SpotLightGPU>() / std::mem::size_of::<f32>(),
            );
        }
        result
    }

    /// Create a new LightUnion from a RectAreaLightGPU.
    pub fn from_rect_area(light: &RectAreaLightGPU, light_id: u32, enabled: bool) -> Self {
        let mut result = Self {
            light_type: LightTypeGPU::RectArea as u32,
            light_id,
            enabled: enabled as u32,
            _pad: 0,
            data: [0.0; LIGHT_DATA_MAX_F32S],
        };
        unsafe {
            let src = light as *const RectAreaLightGPU as *const f32;
            let dst = result.data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(
                src,
                dst,
                std::mem::size_of::<RectAreaLightGPU>() / std::mem::size_of::<f32>(),
            );
        }
        result
    }

    /// Create a new LightUnion from a DiskAreaLightGPU.
    pub fn from_disk_area(light: &DiskAreaLightGPU, light_id: u32, enabled: bool) -> Self {
        let mut result = Self {
            light_type: LightTypeGPU::DiskArea as u32,
            light_id,
            enabled: enabled as u32,
            _pad: 0,
            data: [0.0; LIGHT_DATA_MAX_F32S],
        };
        unsafe {
            let src = light as *const DiskAreaLightGPU as *const f32;
            let dst = result.data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(
                src,
                dst,
                std::mem::size_of::<DiskAreaLightGPU>() / std::mem::size_of::<f32>(),
            );
        }
        result
    }

    /// Create a new LightUnion from an IESLightGPU.
    pub fn from_ies(light: &IESLightGPU, light_id: u32, enabled: bool) -> Self {
        let mut result = Self {
            light_type: LightTypeGPU::IES as u32,
            light_id,
            enabled: enabled as u32,
            _pad: 0,
            data: [0.0; LIGHT_DATA_MAX_F32S],
        };
        unsafe {
            let src = light as *const IESLightGPU as *const f32;
            let dst = result.data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(
                src,
                dst,
                std::mem::size_of::<IESLightGPU>() / std::mem::size_of::<f32>(),
            );
        }
        result
    }

    /// Create a new LightUnion from a SkyLightGPU.
    pub fn from_sky(light: &SkyLightGPU, light_id: u32, enabled: bool) -> Self {
        let mut result = Self {
            light_type: LightTypeGPU::Sky as u32,
            light_id,
            enabled: enabled as u32,
            _pad: 0,
            data: [0.0; LIGHT_DATA_MAX_F32S],
        };
        unsafe {
            let src = light as *const SkyLightGPU as *const f32;
            let dst = result.data.as_mut_ptr();
            std::ptr::copy_nonoverlapping(
                src,
                dst,
                std::mem::size_of::<SkyLightGPU>() / std::mem::size_of::<f32>(),
            );
        }
        result
    }

    /// Get the light type enum value.
    pub fn get_light_type(&self) -> Option<LightTypeGPU> {
        LightTypeGPU::from_u32(self.light_type)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

impl DirectionalLightGPU {
    /// Create a new directional light with default sun-like parameters.
    pub fn new_sun(direction: [f32; 3], color: [f32; 3], intensity: f32) -> Self {
        Self {
            direction,
            angular_diameter: 0.00935, // Sun's angular diameter
            color,
            intensity,
            cascade_count: 4,
            shadow_mode: ShadowModeGPU::Dynamic as u32,
            shadow_bias: 0.001,
            shadow_resolution_scale: 1.0,
            cascade_distances: [10.0, 30.0, 100.0, 500.0],
        }
    }

    /// Calculate approximate luminous power (lumens).
    pub fn luminous_power(&self) -> f32 {
        self.intensity * 1000.0
    }
}

impl PointLightGPU {
    /// Create a new point light with default parameters.
    pub fn new(position: [f32; 3], color: [f32; 3], intensity: f32, radius: f32) -> Self {
        Self {
            position,
            radius,
            color,
            intensity,
            falloff_exponent: 2.0,
            shadow_mode: ShadowModeGPU::Dynamic as u32,
            shadow_bias: 0.001,
            _pad0: 0.0,
        }
    }

    /// Calculate luminous power (lumens) for point light.
    pub fn luminous_power(&self) -> f32 {
        self.intensity * 4.0 * PI
    }

    /// Calculate attenuation at a given distance.
    pub fn attenuation(&self, distance: f32) -> f32 {
        if distance >= self.radius {
            return 0.0;
        }
        let d_over_r = distance / self.radius;
        let numerator = (1.0 - d_over_r.powi(4)).max(0.0).powi(2);
        let denominator = distance.powf(self.falloff_exponent) + 1.0;
        numerator / denominator
    }
}

impl SpotLightGPU {
    /// Create a new spot light with default parameters.
    pub fn new(
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        inner_angle: f32,
        outer_angle: f32,
        radius: f32,
    ) -> Self {
        Self {
            position,
            radius,
            direction,
            intensity,
            color,
            inner_angle,
            outer_angle,
            falloff_exponent: 2.0,
            shadow_mode: ShadowModeGPU::Dynamic as u32,
            shadow_bias: 0.001,
        }
    }

    /// Calculate luminous power for spot light.
    pub fn luminous_power(&self) -> f32 {
        let solid_angle = 2.0 * PI * (1.0 - self.outer_angle.cos());
        self.intensity * solid_angle
    }
}

impl RectAreaLightGPU {
    /// Create a new rectangular area light.
    pub fn new(
        position: [f32; 3],
        direction: [f32; 3],
        up: [f32; 3],
        width: f32,
        height: f32,
        color: [f32; 3],
        intensity: f32,
    ) -> Self {
        Self {
            position,
            width,
            direction,
            height,
            up,
            intensity,
            color,
            two_sided: 0,
            gi_importance: GIImportanceGPU::Medium as u32,
            gi_emissive: 1,
            _pad0: [0, 0],
        }
    }

    /// Calculate the area of the light source.
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Calculate luminous power (nits * area * PI).
    pub fn luminous_power(&self) -> f32 {
        self.intensity * self.area() * PI
    }
}

impl DiskAreaLightGPU {
    /// Create a new disk area light.
    pub fn new(
        position: [f32; 3],
        direction: [f32; 3],
        disk_radius: f32,
        color: [f32; 3],
        intensity: f32,
    ) -> Self {
        Self {
            position,
            disk_radius,
            direction,
            intensity,
            color,
            two_sided: 0,
        }
    }

    /// Calculate the area of the disk.
    pub fn area(&self) -> f32 {
        PI * self.disk_radius * self.disk_radius
    }

    /// Calculate luminous power (nits * area * PI).
    pub fn luminous_power(&self) -> f32 {
        self.intensity * self.area() * PI
    }
}

impl IESLightGPU {
    /// Create a new IES light.
    pub fn new(
        position: [f32; 3],
        direction: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        radius: f32,
        profile_index: u32,
    ) -> Self {
        Self {
            position,
            radius,
            direction,
            intensity,
            color,
            profile_index,
            shadow_mode: ShadowModeGPU::Dynamic as u32,
            shadow_bias: 0.001,
            profile_lumens: 0.0,
            _pad0: 0.0,
        }
    }
}

impl SkyLightGPU {
    /// Create a new sky light.
    pub fn new(color: [f32; 3], intensity: f32, cubemap_index: u32) -> Self {
        Self {
            color,
            intensity,
            lower_hemisphere_color: [0.0, 0.0, 0.0],
            use_lower_hemisphere: 0,
            rotation: 0.0,
            mip_count: 8,
            use_cubemap_for_gi: 1,
            cubemap_index,
        }
    }

    /// Calculate approximate luminous power.
    pub fn luminous_power(&self) -> f32 {
        self.intensity * 10000.0
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_light_type_enum_values() {
        // Verify enum values match expected GPU constants
        assert_eq!(LightTypeGPU::Directional as u32, 0);
        assert_eq!(LightTypeGPU::Point as u32, 1);
        assert_eq!(LightTypeGPU::Spot as u32, 2);
        assert_eq!(LightTypeGPU::RectArea as u32, 3);
        assert_eq!(LightTypeGPU::DiskArea as u32, 4);
        assert_eq!(LightTypeGPU::IES as u32, 5);
        assert_eq!(LightTypeGPU::Sky as u32, 6);
    }

    #[test]
    fn test_light_type_from_u32() {
        assert_eq!(LightTypeGPU::from_u32(0), Some(LightTypeGPU::Directional));
        assert_eq!(LightTypeGPU::from_u32(1), Some(LightTypeGPU::Point));
        assert_eq!(LightTypeGPU::from_u32(6), Some(LightTypeGPU::Sky));
        assert_eq!(LightTypeGPU::from_u32(7), None);
        assert_eq!(LightTypeGPU::from_u32(255), None);
    }

    #[test]
    fn test_directional_light_size_and_alignment() {
        assert_eq!(std::mem::size_of::<DirectionalLightGPU>(), 64);
        // Alignment should be 4 (f32 alignment)
        assert_eq!(std::mem::align_of::<DirectionalLightGPU>(), 4);
        // Size should be 16-byte divisible for GPU uniform buffers
        assert_eq!(std::mem::size_of::<DirectionalLightGPU>() % 16, 0);
    }

    #[test]
    fn test_point_light_size_and_alignment() {
        assert_eq!(std::mem::size_of::<PointLightGPU>(), 48);
        assert_eq!(std::mem::align_of::<PointLightGPU>(), 4);
        assert_eq!(std::mem::size_of::<PointLightGPU>() % 16, 0);
    }

    #[test]
    fn test_spot_light_size_and_alignment() {
        assert_eq!(std::mem::size_of::<SpotLightGPU>(), 64);
        assert_eq!(std::mem::align_of::<SpotLightGPU>(), 4);
        assert_eq!(std::mem::size_of::<SpotLightGPU>() % 16, 0);
    }

    #[test]
    fn test_rect_area_light_size_and_alignment() {
        assert_eq!(std::mem::size_of::<RectAreaLightGPU>(), 80);
        assert_eq!(std::mem::align_of::<RectAreaLightGPU>(), 4);
        assert_eq!(std::mem::size_of::<RectAreaLightGPU>() % 16, 0);
    }

    #[test]
    fn test_disk_area_light_size_and_alignment() {
        assert_eq!(std::mem::size_of::<DiskAreaLightGPU>(), 48);
        assert_eq!(std::mem::align_of::<DiskAreaLightGPU>(), 4);
        assert_eq!(std::mem::size_of::<DiskAreaLightGPU>() % 16, 0);
    }

    #[test]
    fn test_ies_light_size_and_alignment() {
        assert_eq!(std::mem::size_of::<IESLightGPU>(), 64);
        assert_eq!(std::mem::align_of::<IESLightGPU>(), 4);
        assert_eq!(std::mem::size_of::<IESLightGPU>() % 16, 0);
    }

    #[test]
    fn test_sky_light_size_and_alignment() {
        assert_eq!(std::mem::size_of::<SkyLightGPU>(), 48);
        assert_eq!(std::mem::align_of::<SkyLightGPU>(), 4);
        assert_eq!(std::mem::size_of::<SkyLightGPU>() % 16, 0);
    }

    #[test]
    fn test_light_union_size_and_alignment() {
        assert_eq!(std::mem::size_of::<LightUnion>(), 96);
        assert_eq!(std::mem::align_of::<LightUnion>(), 4);
        assert_eq!(std::mem::size_of::<LightUnion>() % 16, 0);
        // Verify union can hold largest light type (RectAreaLightGPU = 80 bytes = 20 f32s)
        assert!(
            LIGHT_DATA_MAX_F32S * std::mem::size_of::<f32>()
                >= std::mem::size_of::<RectAreaLightGPU>()
        );
    }

    #[test]
    fn test_directional_light_creation() {
        let sun = DirectionalLightGPU::new_sun(
            [0.0, -1.0, 0.0],
            [1.0, 0.95, 0.9],
            100000.0,
        );
        assert_eq!(sun.direction, [0.0, -1.0, 0.0]);
        assert_eq!(sun.color, [1.0, 0.95, 0.9]);
        assert_eq!(sun.intensity, 100000.0);
        assert_eq!(sun.cascade_count, 4);
        assert_eq!(sun.shadow_mode, ShadowModeGPU::Dynamic as u32);
    }

    #[test]
    fn test_point_light_attenuation() {
        let light = PointLightGPU::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1000.0, 10.0);

        // At distance 0, attenuation should be 1.0
        let atten_0 = light.attenuation(0.0);
        assert!((atten_0 - 1.0).abs() < 0.01);

        // At radius, attenuation should be 0.0
        let atten_r = light.attenuation(10.0);
        assert_eq!(atten_r, 0.0);

        // Beyond radius, attenuation should be 0.0
        let atten_beyond = light.attenuation(15.0);
        assert_eq!(atten_beyond, 0.0);

        // At half radius, attenuation should be > 0
        let atten_half = light.attenuation(5.0);
        assert!(atten_half > 0.0);
        assert!(atten_half < 1.0);
    }

    #[test]
    fn test_light_union_from_directional() {
        let dir_light = DirectionalLightGPU::new_sun(
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            100000.0,
        );
        let union = LightUnion::from_directional(&dir_light, 42, true);

        assert_eq!(union.light_type, LightTypeGPU::Directional as u32);
        assert_eq!(union.light_id, 42);
        assert_eq!(union.enabled, 1);
        assert_eq!(union.get_light_type(), Some(LightTypeGPU::Directional));

        // Verify data was copied correctly (first 3 floats should be direction)
        assert_eq!(union.data[0], 0.0);
        assert_eq!(union.data[1], -1.0);
        assert_eq!(union.data[2], 0.0);
    }

    #[test]
    fn test_light_union_from_point() {
        let point_light = PointLightGPU::new([1.0, 2.0, 3.0], [1.0, 0.5, 0.0], 800.0, 15.0);
        let union = LightUnion::from_point(&point_light, 7, false);

        assert_eq!(union.light_type, LightTypeGPU::Point as u32);
        assert_eq!(union.light_id, 7);
        assert_eq!(union.enabled, 0);
        assert_eq!(union.get_light_type(), Some(LightTypeGPU::Point));

        // Verify position was copied
        assert_eq!(union.data[0], 1.0);
        assert_eq!(union.data[1], 2.0);
        assert_eq!(union.data[2], 3.0);
        assert_eq!(union.data[3], 15.0); // radius
    }

    #[test]
    fn test_spot_light_luminous_power() {
        let spot = SpotLightGPU::new(
            [0.0, 5.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            1000.0,             // candelas
            0.4363,             // ~25 degrees inner
            0.7854,             // ~45 degrees outer
            20.0,
        );
        let power = spot.luminous_power();
        // Solid angle = 2*PI*(1-cos(45deg)) ~ 1.84
        // Power = 1000 * 1.84 ~ 1840
        assert!(power > 1500.0);
        assert!(power < 2500.0);
    }

    #[test]
    fn test_rect_area_light_area_and_power() {
        let rect = RectAreaLightGPU::new(
            [0.0, 3.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            2.0,    // width
            1.0,    // height
            [1.0, 1.0, 1.0],
            500.0,  // nits
        );
        assert_eq!(rect.area(), 2.0);
        // Power = 500 * 2 * PI ~ 3141.6
        let power = rect.luminous_power();
        assert!((power - 500.0 * 2.0 * PI).abs() < 0.01);
    }

    #[test]
    fn test_disk_area_light_area_and_power() {
        let disk = DiskAreaLightGPU::new(
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 0.0],
            0.5,    // radius
            [1.0, 0.8, 0.6],
            1000.0, // nits
        );
        // Area = PI * 0.5^2 = PI * 0.25
        let expected_area = PI * 0.25;
        assert!((disk.area() - expected_area).abs() < 0.0001);
        // Power = 1000 * PI*0.25 * PI
        let power = disk.luminous_power();
        assert!((power - 1000.0 * expected_area * PI).abs() < 0.01);
    }

    #[test]
    fn test_all_light_types_fit_in_union() {
        // Ensure all light types can fit in the union's data buffer
        let union_data_bytes = LIGHT_DATA_MAX_F32S * std::mem::size_of::<f32>();
        assert!(std::mem::size_of::<DirectionalLightGPU>() <= union_data_bytes);
        assert!(std::mem::size_of::<PointLightGPU>() <= union_data_bytes);
        assert!(std::mem::size_of::<SpotLightGPU>() <= union_data_bytes);
        assert!(std::mem::size_of::<RectAreaLightGPU>() <= union_data_bytes);
        assert!(std::mem::size_of::<DiskAreaLightGPU>() <= union_data_bytes);
        assert!(std::mem::size_of::<IESLightGPU>() <= union_data_bytes);
        assert!(std::mem::size_of::<SkyLightGPU>() <= union_data_bytes);
    }

    #[test]
    fn test_shadow_mode_enum_values() {
        assert_eq!(ShadowModeGPU::None as u32, 0);
        assert_eq!(ShadowModeGPU::Static as u32, 1);
        assert_eq!(ShadowModeGPU::Dynamic as u32, 2);
    }

    #[test]
    fn test_gi_importance_enum_values() {
        assert_eq!(GIImportanceGPU::Low as u32, 0);
        assert_eq!(GIImportanceGPU::Medium as u32, 1);
        assert_eq!(GIImportanceGPU::High as u32, 2);
        assert_eq!(GIImportanceGPU::Critical as u32, 3);
    }
}
