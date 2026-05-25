// Vector types.
//
// FVec* -- Fixed32 vectors for deterministic ECS transforms.
// Vec*  -- f32 vectors for the rendering path.
//
// All types derive bytemuck Pod/Zeroable for GPU upload.

use crate::fixed::Fixed32;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
// FVec2
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct FVec2 {
    pub x: Fixed32,
    pub y: Fixed32,
}

impl FVec2 {
    pub const ZERO: Self = Self { x: Fixed32::ZERO, y: Fixed32::ZERO };
    pub const UNIT_X: Self = Self { x: Fixed32::ONE, y: Fixed32::ZERO };
    pub const UNIT_Y: Self = Self { x: Fixed32::ZERO, y: Fixed32::ONE };

    #[inline]
    pub const fn new(x: Fixed32, y: Fixed32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn dot(self, other: Self) -> Fixed32 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn length_squared(self) -> Fixed32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> Fixed32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == Fixed32::ZERO {
            return Self::ZERO;
        }
        self / len
    }

    #[inline]
    pub fn lerp(self, other: Self, t: Fixed32) -> Self {
        self + (other - self) * t
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self { x: if self.x < other.x { self.x } else { other.x }, y: if self.y < other.y { self.y } else { other.y } }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self { x: if self.x > other.x { self.x } else { other.x }, y: if self.y > other.y { self.y } else { other.y } }
    }

    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self { x: if self.x < min.x { min.x } else if self.x > max.x { max.x } else { self.x }, y: if self.y < min.y { min.y } else if self.y > max.y { max.y } else { self.y } }
    }
}

impl Add for FVec2 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
}

impl AddAssign for FVec2 {
    #[inline]
    fn add_assign(&mut self, other: Self) { *self = *self + other; }
}

impl Sub for FVec2 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }
}

impl SubAssign for FVec2 {
    #[inline]
    fn sub_assign(&mut self, other: Self) { *self = *self - other; }
}

impl Mul<Fixed32> for FVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, s: Fixed32) -> Self {
        Self { x: self.x * s, y: self.y * s }
    }
}

impl MulAssign<Fixed32> for FVec2 {
    #[inline]
    fn mul_assign(&mut self, s: Fixed32) { *self = *self * s; }
}

impl Div<Fixed32> for FVec2 {
    type Output = Self;
    #[inline]
    fn div(self, s: Fixed32) -> Self {
        Self { x: self.x / s, y: self.y / s }
    }
}

impl DivAssign<Fixed32> for FVec2 {
    #[inline]
    fn div_assign(&mut self, s: Fixed32) { *self = *self / s; }
}

impl Neg for FVec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y } }
}

// SAFETY: FVec2 is repr(C) with two Fixed32 fields.
unsafe impl bytemuck::Zeroable for FVec2 {}
unsafe impl bytemuck::Pod for FVec2 {}

// ---------------------------------------------------------------------------
// FVec3
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct FVec3 {
    pub x: Fixed32,
    pub y: Fixed32,
    pub z: Fixed32,
}

impl FVec3 {
    pub const ZERO: Self = Self { x: Fixed32::ZERO, y: Fixed32::ZERO, z: Fixed32::ZERO };
    pub const UNIT_X: Self = Self { x: Fixed32::ONE, y: Fixed32::ZERO, z: Fixed32::ZERO };
    pub const UNIT_Y: Self = Self { x: Fixed32::ZERO, y: Fixed32::ONE, z: Fixed32::ZERO };
    pub const UNIT_Z: Self = Self { x: Fixed32::ZERO, y: Fixed32::ZERO, z: Fixed32::ONE };

    #[inline]
    pub const fn new(x: Fixed32, y: Fixed32, z: Fixed32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, other: Self) -> Fixed32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn length_squared(self) -> Fixed32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> Fixed32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == Fixed32::ZERO {
            return Self::ZERO;
        }
        self / len
    }

    #[inline]
    pub fn lerp(self, other: Self, t: Fixed32) -> Self {
        self + (other - self) * t
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self { x: if self.x < other.x { self.x } else { other.x }, y: if self.y < other.y { self.y } else { other.y }, z: if self.z < other.z { self.z } else { other.z } }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self { x: if self.x > other.x { self.x } else { other.x }, y: if self.y > other.y { self.y } else { other.y }, z: if self.z > other.z { self.z } else { other.z } }
    }
}

impl Add for FVec3 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z } }
}

impl AddAssign for FVec3 {
    #[inline]
    fn add_assign(&mut self, other: Self) { *self = *self + other; }
}

impl Sub for FVec3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self { Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z } }
}

impl SubAssign for FVec3 {
    #[inline]
    fn sub_assign(&mut self, other: Self) { *self = *self - other; }
}

impl Mul<Fixed32> for FVec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: Fixed32) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}

impl MulAssign<Fixed32> for FVec3 {
    #[inline]
    fn mul_assign(&mut self, s: Fixed32) { *self = *self * s; }
}

impl Div<Fixed32> for FVec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: Fixed32) -> Self { Self { x: self.x / s, y: self.y / s, z: self.z / s } }
}

impl DivAssign<Fixed32> for FVec3 {
    #[inline]
    fn div_assign(&mut self, s: Fixed32) { *self = *self / s; }
}

impl Neg for FVec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

// SAFETY: FVec3 is repr(C) with three Fixed32 fields.
unsafe impl bytemuck::Zeroable for FVec3 {}
unsafe impl bytemuck::Pod for FVec3 {}

// ---------------------------------------------------------------------------
// FVec4
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct FVec4 {
    pub x: Fixed32,
    pub y: Fixed32,
    pub z: Fixed32,
    pub w: Fixed32,
}

impl FVec4 {
    pub const ZERO: Self = Self { x: Fixed32::ZERO, y: Fixed32::ZERO, z: Fixed32::ZERO, w: Fixed32::ZERO };
    pub const UNIT_X: Self = Self { x: Fixed32::ONE, y: Fixed32::ZERO, z: Fixed32::ZERO, w: Fixed32::ZERO };
    pub const UNIT_Y: Self = Self { x: Fixed32::ZERO, y: Fixed32::ONE, z: Fixed32::ZERO, w: Fixed32::ZERO };
    pub const UNIT_Z: Self = Self { x: Fixed32::ZERO, y: Fixed32::ZERO, z: Fixed32::ONE, w: Fixed32::ZERO };
    pub const UNIT_W: Self = Self { x: Fixed32::ZERO, y: Fixed32::ZERO, z: Fixed32::ZERO, w: Fixed32::ONE };

    #[inline]
    pub const fn new(x: Fixed32, y: Fixed32, z: Fixed32, w: Fixed32) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn dot(self, other: Self) -> Fixed32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    #[inline]
    pub fn length_squared(self) -> Fixed32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> Fixed32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn lerp(self, other: Self, t: Fixed32) -> Self {
        self + (other - self) * t
    }
}

impl Add for FVec4 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z, w: self.w + other.w } }
}

impl Sub for FVec4 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self { Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z, w: self.w - other.w } }
}

impl Mul<Fixed32> for FVec4 {
    type Output = Self;
    #[inline]
    fn mul(self, s: Fixed32) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s, w: self.w * s } }
}

impl Div<Fixed32> for FVec4 {
    type Output = Self;
    #[inline]
    fn div(self, s: Fixed32) -> Self { Self { x: self.x / s, y: self.y / s, z: self.z / s, w: self.w / s } }
}

// SAFETY: FVec4 is repr(C) with four Fixed32 fields.
unsafe impl bytemuck::Zeroable for FVec4 {}
unsafe impl bytemuck::Pod for FVec4 {}

// ===========================================================================
// Vec2 (f32)
// ===========================================================================

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0 };
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0 };

    #[inline]
    pub const fn new(x: f32, y: f32) -> Self { Self { x, y } }

    #[inline]
    pub fn dot(self, other: Self) -> f32 { self.x * other.x + self.y * other.y }

    #[inline]
    pub fn length_squared(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 { self.length_squared().sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == 0.0 { return Self::ZERO; }
        Self { x: self.x / len, y: self.y / len }
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self { x: self.x + (other.x - self.x) * t, y: self.y + (other.y - self.y) * t }
    }
}

impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y } }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self { Self { x: self.x - other.x, y: self.y - other.y } }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self { Self { x: self.x * s, y: self.y * s } }
}

unsafe impl bytemuck::Zeroable for Vec2 {}
unsafe impl bytemuck::Pod for Vec2 {}

// ===========================================================================
// Vec3 (f32)
// ===========================================================================

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const UNIT_Z: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    #[inline]
    pub fn dot(self, other: Self) -> f32 { self.x * other.x + self.y * other.y + self.z * other.z }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn length_squared(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 { self.length_squared().sqrt() }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == 0.0 { return Self::ZERO; }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self { x: self.x + (other.x - self.x) * t, y: self.y + (other.y - self.y) * t, z: self.z + (other.z - self.z) * t }
    }

    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z } }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self { Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z } }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}

unsafe impl bytemuck::Zeroable for Vec3 {}
unsafe impl bytemuck::Pod for Vec3 {}

// ===========================================================================
// Vec4 (f32)
// ===========================================================================

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0, z: 0.0, w: 0.0 };
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0, z: 0.0, w: 0.0 };
    pub const UNIT_Z: Self = Self { x: 0.0, y: 0.0, z: 1.0, w: 0.0 };
    pub const UNIT_W: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self { Self { x, y, z, w } }

    #[inline]
    pub fn dot(self, other: Self) -> f32 { self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w }

    #[inline]
    pub fn length_squared(self) -> f32 { self.dot(self) }

    #[inline]
    pub fn length(self) -> f32 { self.length_squared().sqrt() }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self { x: self.x + (other.x - self.x) * t, y: self.y + (other.y - self.y) * t, z: self.z + (other.z - self.z) * t, w: self.w + (other.w - self.w) * t }
    }
}

impl Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z, w: self.w + other.w } }
}

impl Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self { Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z, w: self.w - other.w } }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s, w: self.w * s } }
}

impl Neg for Vec4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z, w: -self.w } }
}

unsafe impl bytemuck::Zeroable for Vec4 {}
unsafe impl bytemuck::Pod for Vec4 {}
