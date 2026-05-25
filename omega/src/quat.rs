// Quaternion types.
//
// FQuat -- fixed-point quaternion (w,x,y,z as Fixed32) for deterministic
//          transforms in the ECS transform pipeline.
// Quat  -- standard f32 quaternion for the rendering path.
//
// Both types derive bytemuck Pod/Zeroable for GPU upload.

use crate::fixed::Fixed32;
use crate::vec::{FVec3, Vec3};
use core::ops::{Mul, MulAssign, Neg};

// ---------------------------------------------------------------------------
// FQuat
// ---------------------------------------------------------------------------

/// Fixed-point quaternion (w,x,y,z components as Fixed32).
///
/// Stored as (w, x, y, z) with the scalar part first.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct FQuat {
    pub w: Fixed32,
    pub x: Fixed32,
    pub y: Fixed32,
    pub z: Fixed32,
}

impl FQuat {
    pub const IDENTITY: Self = Self { w: Fixed32::ONE, x: Fixed32::ZERO, y: Fixed32::ZERO, z: Fixed32::ZERO };
    pub const SLERP_THRESHOLD: f64 = 0.9995;

    #[inline]
    pub const fn new(w: Fixed32, x: Fixed32, y: Fixed32, z: Fixed32) -> Self {
        Self { w, x, y, z }
    }

    #[inline]
    pub fn from_axis_angle(axis: FVec3, angle: Fixed32) -> Self {
        let half = angle / Fixed32::from_f32(2.0);
        let (s, c) = half.sin_cos_approx();
        let scaled = axis * s;
        Self { w: c, x: scaled.x, y: scaled.y, z: scaled.z }
    }

    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }

    #[inline]
    pub fn length_squared(self) -> Fixed32 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    pub fn length(self) -> Fixed32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq == Fixed32::ZERO {
            return Self::IDENTITY;
        }
        let conj = self.conjugate();
        Self { w: conj.w / len_sq, x: conj.x / len_sq, y: conj.y / len_sq, z: conj.z / len_sq }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == Fixed32::ZERO {
            return Self::IDENTITY;
        }
        Self { w: self.w / len, x: self.x / len, y: self.y / len, z: self.z / len }
    }

    #[inline]
    pub fn rotate_vector(self, v: FVec3) -> FVec3 {
        let qv = FVec3::new(self.x, self.y, self.z);
        let t = qv.cross(v) * Fixed32::from_f32(2.0);
        v + qv.cross(t) + t * self.w
    }

    /// Spherical linear interpolation.
    #[inline]
    pub fn slerp(self, other: Self, t: Fixed32) -> Self {
        let dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let other = if dot.to_f64() < 0.0 { -other } else { other };
        let dot = if dot.to_f64() < 0.0 { -dot } else { dot };

        if dot.to_f64() > Self::SLERP_THRESHOLD {
            let mut result = Self {
                w: self.w + (other.w - self.w) * t,
                x: self.x + (other.x - self.x) * t,
                y: self.y + (other.y - self.y) * t,
                z: self.z + (other.z - self.z) * t,
            };
            let len = result.length();
            if len != Fixed32::ZERO {
                result.w = result.w / len;
                result.x = result.x / len;
                result.y = result.y / len;
                result.z = result.z / len;
            }
            return result;
        }

        let theta = Fixed32::from_f64(dot.to_f64().acos());
        let sin_theta = theta.sin_approx();
        if sin_theta == Fixed32::ZERO {
            return self;
        }

        let a = (theta * (Fixed32::ONE - t)).sin_approx() / sin_theta;
        let b = (theta * t).sin_approx() / sin_theta;

        Self {
            w: self.w * a + other.w * b,
            x: self.x * a + other.x * b,
            y: self.y * a + other.y * b,
            z: self.z * a + other.z * b,
        }
    }
}

impl Mul for FQuat {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self { self.mul(other) }
}

impl MulAssign for FQuat {
    #[inline]
    fn mul_assign(&mut self, other: Self) { *self = *self * other; }
}

impl Neg for FQuat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { Self { w: -self.w, x: -self.x, y: -self.y, z: -self.z } }
}

// SAFETY: FQuat is repr(C) with four Fixed32 fields.
unsafe impl bytemuck::Zeroable for FQuat {}
unsafe impl bytemuck::Pod for FQuat {}

// ===========================================================================
// Quat (f32)
// ===========================================================================

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Quat {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

const QUAT_SLERP_THRESHOLD: f32 = 0.9995;

impl Quat {
    pub const IDENTITY: Self = Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self { Self { w, x, y, z } }

    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half = angle * 0.5;
        let (s, c) = half.sin_cos();
        Self { w: c, x: axis.x * s, y: axis.y * s, z: axis.z * s }
    }

    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    #[inline]
    pub fn conjugate(self) -> Self { Self { w: self.w, x: -self.x, y: -self.y, z: -self.z } }

    #[inline]
    pub fn length_squared(self) -> f32 { self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z }

    #[inline]
    pub fn length(self) -> f32 { self.length_squared().sqrt() }

    #[inline]
    pub fn inverse(self) -> Self {
        let ls = self.length_squared();
        if ls == 0.0 { return Self::IDENTITY; }
        let c = self.conjugate();
        Self { w: c.w / ls, x: c.x / ls, y: c.y / ls, z: c.z / ls }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len == 0.0 { return Self::IDENTITY; }
        Self { w: self.w / len, x: self.x / len, y: self.y / len, z: self.z / len }
    }

    #[inline]
    pub fn rotate_vector(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t = qv.cross(v) * 2.0;
        v + qv.cross(t) + t * self.w
    }

    #[inline]
    pub fn slerp(self, other: Self, t: f32) -> Self {
        let mut dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let mut other = other;
        if dot < 0.0 {
            dot = -dot;
            other = -other;
        }
        if dot > QUAT_SLERP_THRESHOLD {
            let mut r = Quat {
                w: self.w + (other.w - self.w) * t,
                x: self.x + (other.x - self.x) * t,
                y: self.y + (other.y - self.y) * t,
                z: self.z + (other.z - self.z) * t,
            };
            let len = r.length();
            if len != 0.0 {
                r.w /= len;
                r.x /= len;
                r.y /= len;
                r.z /= len;
            }
            return r;
        }
        let theta = dot.acos();
        let sin_theta = theta.sin();
        if sin_theta == 0.0 { return self; }
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;
        Self { w: self.w * a + other.w * b, x: self.x * a + other.x * b, y: self.y * a + other.y * b, z: self.z * a + other.z * b }
    }
}

impl Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self { self.mul(other) }
}

impl MulAssign for Quat {
    #[inline]
    fn mul_assign(&mut self, other: Self) { *self = *self * other; }
}

impl Neg for Quat {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { Self { w: -self.w, x: -self.x, y: -self.y, z: -self.z } }
}

// SAFETY: Quat is repr(C) with four f32 fields.
unsafe impl bytemuck::Zeroable for Quat {}
unsafe impl bytemuck::Pod for Quat {}
