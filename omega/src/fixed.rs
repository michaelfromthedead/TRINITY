// Fixed-point number types for deterministic arithmetic.
//
// Fixed16 -- Q8.8 (i16 scale 256) for smaller state.
// Fixed32 -- Q16.16 (i32 scale 65536) for ECS transforms.
//
// Operations saturate on overflow rather than wrapping or panicking,
// ensuring the engine stays in a well-defined state.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

// ---------------------------------------------------------------------------
// Fixed16 (Q8.8)
// ---------------------------------------------------------------------------

/// Q8.8 fixed-point number (i16 backing, scale 256).
///
/// Range: [-128.0, 127.99609375]
/// Precision: 0.00390625
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Fixed16(pub i16);

impl Fixed16 {
    pub const SCALE: i32 = 256;
    pub const SCALE_F64: f64 = 256.0;

    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(256);
    pub const MIN: Self = Self(i16::MIN);
    pub const MAX: Self = Self(i16::MAX);
    pub const EPSILON: Self = Self(1); // 1/256

    #[inline]
    pub const fn from_raw(v: i16) -> Self {
        Self(v)
    }

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let scaled = (val * Self::SCALE as f32).round() as i32;
        if scaled > i16::MAX as i32 {
            Self(i16::MAX)
        } else if scaled < i16::MIN as i32 {
            Self(i16::MIN)
        } else {
            Self(scaled as i16)
        }
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }

    #[inline]
    pub fn from_f64(val: f64) -> Self {
        let scaled = (val * Self::SCALE_F64).round() as i32;
        if scaled > i16::MAX as i32 {
            Self(i16::MAX)
        } else if scaled < i16::MIN as i32 {
            Self(i16::MIN)
        } else {
            Self(scaled as i16)
        }
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE_F64
    }

    /// Saturating addition.
    #[inline]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Saturating subtraction.
    #[inline]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Saturating multiplication (in Q8.8 space).
    #[inline]
    pub fn saturating_mul(self, other: Self) -> Self {
        let a = self.0 as i32;
        let b = other.0 as i32;
        let product = a * b;
        let result = product / Self::SCALE;
        if result > i16::MAX as i32 {
            Self(i16::MAX)
        } else if result < i16::MIN as i32 {
            Self(i16::MIN)
        } else {
            Self(result as i16)
        }
    }

    /// Saturating division (in Q8.8 space).
    #[inline]
    pub fn saturating_div(self, other: Self) -> Self {
        if other.0 == 0 {
            return if self.is_negative() {
                Self(i16::MIN)
            } else {
                Self(i16::MAX)
            };
        }
        let a = self.0 as i32;
        let b = other.0 as i32;
        let result = (a * Self::SCALE) / b;
        if result > i16::MAX as i32 {
            Self(i16::MAX)
        } else if result < i16::MIN as i32 {
            Self(i16::MIN)
        } else {
            Self(result as i16)
        }
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.saturating_abs())
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Floor (round towards negative infinity).
    #[inline]
    pub fn floor(self) -> Self {
        if self.0 >= 0 {
            Self((self.0 / Self::SCALE as i16) * Self::SCALE as i16)
        } else {
            Self(((self.0 - Self::SCALE as i16 + 1) / Self::SCALE as i16) * Self::SCALE as i16)
        }
    }

    /// Ceil (round towards positive infinity).
    #[inline]
    pub fn ceil(self) -> Self {
        if self.0 <= 0 {
            Self((self.0 / Self::SCALE as i16) * Self::SCALE as i16)
        } else {
            Self(((self.0 + Self::SCALE as i16 - 1) / Self::SCALE as i16) * Self::SCALE as i16)
        }
    }

    /// Round towards zero (truncation).
    #[inline]
    pub fn round_to_zero(self) -> i16 {
        self.0 / Self::SCALE as i16
    }

    /// Linear interpolation.
    #[inline]
    pub fn lerp(self, other: Self, t: Self) -> Self {
        self + (other - self) * t
    }
}

impl Add for Fixed16 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl AddAssign for Fixed16 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for Fixed16 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
}

impl SubAssign for Fixed16 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Mul for Fixed16 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        self.saturating_mul(other)
    }
}

impl MulAssign for Fixed16 {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl Div for Fixed16 {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        self.saturating_div(other)
    }
}

impl DivAssign for Fixed16 {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl Neg for Fixed16 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0.saturating_neg())
    }
}

// -- Conversions ----------------------------------------------------------

impl From<i16> for Fixed16 {
    #[inline]
    fn from(v: i16) -> Self {
        let scaled = v as i32 * Self::SCALE;
        if scaled > i16::MAX as i32 {
            Self(i16::MAX)
        } else if scaled < i16::MIN as i32 {
            Self(i16::MIN)
        } else {
            Self(scaled as i16)
        }
    }
}

impl From<f32> for Fixed16 {
    #[inline]
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<Fixed16> for f32 {
    #[inline]
    fn from(v: Fixed16) -> Self {
        v.to_f32()
    }
}

impl From<Fixed16> for f64 {
    #[inline]
    fn from(v: Fixed16) -> Self {
        v.to_f64()
    }
}

// -- bytemuck -------------------------------------------------------------

// SAFETY: Fixed16 is a transparent wrapper around i16.
unsafe impl bytemuck::Zeroable for Fixed16 {}
unsafe impl bytemuck::Pod for Fixed16 {}

// -- serde -----------------------------------------------------------------

#[cfg(feature = "serde")]
impl serde::Serialize for Fixed16 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_f32().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Fixed16 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        f32::deserialize(deserializer).map(Self::from_f32)
    }
}

// ===========================================================================
// Fixed32 (Q16.16)
// ===========================================================================

/// Q16.16 fixed-point number (i32 backing, scale 65536).
///
/// Range: [-32768.0, 32767.99998474]
/// Precision: ~0.00001526
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Fixed32(pub i32);

impl Fixed32 {
    pub const SCALE: i64 = 65536;
    pub const SCALE_F64: f64 = 65536.0;

    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(65536);
    pub const MIN: Self = Self(i32::MIN);
    pub const MAX: Self = Self(i32::MAX);
    pub const EPSILON: Self = Self(1); // 1/65536

    #[inline]
    pub const fn from_raw(v: i32) -> Self {
        Self(v)
    }

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let scaled = (val * Self::SCALE as f32).round() as i64;
        if scaled > i32::MAX as i64 {
            Self(i32::MAX)
        } else if scaled < i32::MIN as i64 {
            Self(i32::MIN)
        } else {
            Self(scaled as i32)
        }
    }

    #[inline]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 / Self::SCALE_F64 as f32
    }

    #[inline]
    pub fn from_f64(val: f64) -> Self {
        let scaled = (val * Self::SCALE_F64).round() as i64;
        if scaled >= i32::MAX as i64 {
            Self(i32::MAX)
        } else if scaled <= i32::MIN as i64 {
            Self(i32::MIN)
        } else {
            Self(scaled as i32)
        }
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE_F64
    }

    /// Saturating addition.
    #[inline]
    pub fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Saturating subtraction.
    #[inline]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Saturating multiplication (in Q16.16 space).
    #[inline]
    pub fn saturating_mul(self, other: Self) -> Self {
        let a = self.0 as i64;
        let b = other.0 as i64;
        let product = a * b;
        let result = product / Self::SCALE;
        if result > i32::MAX as i64 {
            Self(i32::MAX)
        } else if result < i32::MIN as i64 {
            Self(i32::MIN)
        } else {
            Self(result as i32)
        }
    }

    /// Saturating division (in Q16.16 space).
    #[inline]
    pub fn saturating_div(self, other: Self) -> Self {
        if other.0 == 0 {
            return if self.is_negative() {
                Self(i32::MIN)
            } else {
                Self(i32::MAX)
            };
        }
        let a = self.0 as i64;
        let b = other.0 as i64;
        let result = (a * Self::SCALE) / b;
        if result > i32::MAX as i64 {
            Self(i32::MAX)
        } else if result < i32::MIN as i64 {
            Self(i32::MIN)
        } else {
            Self(result as i32)
        }
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.saturating_abs())
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }

    /// Floor (round towards negative infinity).
    #[inline]
    pub fn floor(self) -> Self {
        let scale = Self::SCALE as i32;
        if self.0 >= 0 {
            Self((self.0 / scale) * scale)
        } else {
            Self(((self.0 - scale + 1) / scale) * scale)
        }
    }

    /// Ceil (round towards positive infinity).
    #[inline]
    pub fn ceil(self) -> Self {
        let scale = Self::SCALE as i32;
        if self.0 <= 0 {
            Self((self.0 / scale) * scale)
        } else {
            Self(((self.0 + scale - 1) / scale) * scale)
        }
    }

    /// Round towards zero (truncation).
    #[inline]
    pub fn round_to_zero(self) -> i32 {
        self.0 / Self::SCALE as i32
    }

    /// Linear interpolation.
    #[inline]
    pub fn lerp(self, other: Self, t: Self) -> Self {
        self + (other - self) * t
    }

    /// Square root (via f64).
    #[inline]
    pub fn sqrt(self) -> Self {
        if self <= Self::ZERO {
            return Self::ZERO;
        }
        Self::from_f64(self.to_f64().sqrt())
    }

    /// Sin approximation (via TrigLUT).
    #[inline]
    pub fn sin_approx(self) -> Self {
        // Delegate to TrigLUT via f32 conversion for now.
        // For a fully fixed-point TrigLUT, use trig::TrigLUT directly.
        let val = self.to_f32();
        Self::from_f32(crate::trig::TrigLUT::sin(val))
    }

    /// Cos approximation (via TrigLUT).
    #[inline]
    pub fn cos_approx(self) -> Self {
        let val = self.to_f32();
        Self::from_f32(crate::trig::TrigLUT::cos(val))
    }

    /// Simultaneous sin and cos approximation (via TrigLUT).
    #[inline]
    pub fn sin_cos_approx(self) -> (Self, Self) {
        let val = self.to_f32();
        let s = crate::trig::TrigLUT::sin(val);
        let c = crate::trig::TrigLUT::cos(val);
        (Self::from_f32(s), Self::from_f32(c))
    }

    /// Tan approximation (via TrigLUT).
    #[inline]
    pub fn tan_approx(self) -> Self {
        let val = self.to_f32();
        Self::from_f32(crate::trig::TrigLUT::tan(val))
    }
}

// -- Arithmetic ------------------------------------------------------------

impl Add for Fixed32 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl AddAssign for Fixed32 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for Fixed32 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
}

impl SubAssign for Fixed32 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Mul for Fixed32 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        self.saturating_mul(other)
    }
}

impl MulAssign for Fixed32 {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl Div for Fixed32 {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        self.saturating_div(other)
    }
}

impl DivAssign for Fixed32 {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl Neg for Fixed32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0.saturating_neg())
    }
}

impl Rem for Fixed32 {
    type Output = Self;
    #[inline]
    fn rem(self, other: Self) -> Self {
        if other.0 == 0 {
            return Self::ZERO;
        }
        Self(self.0 % other.0)
    }
}

impl RemAssign for Fixed32 {
    #[inline]
    fn rem_assign(&mut self, other: Self) {
        *self = *self % other;
    }
}

// -- Conversions ----------------------------------------------------------

impl From<i32> for Fixed32 {
    #[inline]
    fn from(v: i32) -> Self {
        let scaled = v as i64 * Self::SCALE;
        if scaled > i32::MAX as i64 {
            Self(i32::MAX)
        } else if scaled < i32::MIN as i64 {
            Self(i32::MIN)
        } else {
            Self(scaled as i32)
        }
    }
}

impl From<Fixed16> for Fixed32 {
    #[inline]
    fn from(v: Fixed16) -> Self {
        // Fixed16 is Q8.8, Fixed32 is Q16.16. Scale by 256.
        Self(v.0 as i32 * 256)
    }
}

impl From<f32> for Fixed32 {
    #[inline]
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<Fixed32> for f32 {
    #[inline]
    fn from(v: Fixed32) -> Self {
        v.to_f32()
    }
}

impl From<Fixed32> for f64 {
    #[inline]
    fn from(v: Fixed32) -> Self {
        v.to_f64()
    }
}

// -- bytemuck -------------------------------------------------------------

// SAFETY: Fixed32 is a transparent wrapper around i32.
unsafe impl bytemuck::Zeroable for Fixed32 {}
unsafe impl bytemuck::Pod for Fixed32 {}

// -- serde -----------------------------------------------------------------

#[cfg(feature = "serde")]
impl serde::Serialize for Fixed32 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_f32().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Fixed32 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        f32::deserialize(deserializer).map(Self::from_f32)
    }
}
