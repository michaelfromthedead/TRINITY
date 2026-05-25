// Matrix types.
//
// M64  -- 4x4 column-major Fixed32 matrix (deterministic, ECS transform).
// Mat3 -- 3x3 column-major f32 matrix (rendering, normal computation).
// Mat4 -- 4x4 column-major f32 matrix (rendering, projection).
//
// All types derive bytemuck Pod/Zeroable for GPU upload.
// Column-major layout matches WGSL mat4x4<f32> when uploaded as uniform.

use crate::fixed::Fixed32;
use crate::vec::{FVec3, FVec4, Vec3, Vec4};
use core::ops::Mul;

// ---------------------------------------------------------------------------
// M64 (4x4 column-major Fixed32)
// ---------------------------------------------------------------------------

/// 4x4 column-major fixed-point matrix.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct M64 {
    pub c0: FVec4,
    pub c1: FVec4,
    pub c2: FVec4,
    pub c3: FVec4,
}

impl M64 {
    pub const IDENTITY: Self = Self {
        c0: FVec4 { x: Fixed32::ONE, y: Fixed32::ZERO, z: Fixed32::ZERO, w: Fixed32::ZERO },
        c1: FVec4 { x: Fixed32::ZERO, y: Fixed32::ONE, z: Fixed32::ZERO, w: Fixed32::ZERO },
        c2: FVec4 { x: Fixed32::ZERO, y: Fixed32::ZERO, z: Fixed32::ONE, w: Fixed32::ZERO },
        c3: FVec4 { x: Fixed32::ZERO, y: Fixed32::ZERO, z: Fixed32::ZERO, w: Fixed32::ONE },
    };

    pub const ZERO: Self = Self {
        c0: FVec4::ZERO, c1: FVec4::ZERO, c2: FVec4::ZERO, c3: FVec4::ZERO,
    };

    #[inline]
    pub const fn from_columns(c0: FVec4, c1: FVec4, c2: FVec4, c3: FVec4) -> Self {
        Self { c0, c1, c2, c3 }
    }

    #[inline]
    pub fn from_col_array(arr: &[Fixed32; 16]) -> Self {
        Self {
            c0: FVec4::new(arr[0], arr[1], arr[2], arr[3]),
            c1: FVec4::new(arr[4], arr[5], arr[6], arr[7]),
            c2: FVec4::new(arr[8], arr[9], arr[10], arr[11]),
            c3: FVec4::new(arr[12], arr[13], arr[14], arr[15]),
        }
    }

    #[inline]
    pub fn get(self, col: usize, row: usize) -> Fixed32 {
        let c = match col { 0 => self.c0, 1 => self.c1, 2 => self.c2, 3 => self.c3, _ => panic!("col") };
        match row { 0 => c.x, 1 => c.y, 2 => c.z, 3 => c.w, _ => panic!("row") }
    }

    #[inline]
    pub fn get_col(self, col: usize) -> FVec4 {
        match col { 0 => self.c0, 1 => self.c1, 2 => self.c2, 3 => self.c3, _ => panic!("col") }
    }

    fn set(&mut self, col: usize, row: usize, val: Fixed32) {
        let c = match col { 0 => &mut self.c0, 1 => &mut self.c1, 2 => &mut self.c2, 3 => &mut self.c3, _ => panic!("col") };
        match row { 0 => c.x = val, 1 => c.y = val, 2 => c.z = val, 3 => c.w = val, _ => panic!("row") }
    }

    /// Matrix-matrix multiply.
    #[inline]
    pub fn mul_m(self, other: Self) -> Self {
        let mut r = Self::ZERO;
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = Fixed32::ZERO;
                for k in 0..4 {
                    sum = sum + self.get(k, row) * other.get(col, k);
                }
                r.set(col, row, sum);
            }
        }
        r
    }

    /// Matrix-vector multiply.
    #[inline]
    pub fn mul_v(self, v: FVec4) -> FVec4 {
        FVec4 {
            x: self.c0.x * v.x + self.c1.x * v.y + self.c2.x * v.z + self.c3.x * v.w,
            y: self.c0.y * v.x + self.c1.y * v.y + self.c2.y * v.z + self.c3.y * v.w,
            z: self.c0.z * v.x + self.c1.z * v.y + self.c2.z * v.z + self.c3.z * v.w,
            w: self.c0.w * v.x + self.c1.w * v.y + self.c2.w * v.z + self.c3.w * v.w,
        }
    }

    /// Matrix-vector multiply for Vec3 (w=1).
    #[inline]
    pub fn mul_v3(self, v: FVec3) -> FVec3 {
        let r = self.mul_v(FVec4::new(v.x, v.y, v.z, Fixed32::ONE));
        FVec3::new(r.x, r.y, r.z)
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            c0: FVec4::new(self.c0.x, self.c1.x, self.c2.x, self.c3.x),
            c1: FVec4::new(self.c0.y, self.c1.y, self.c2.y, self.c3.y),
            c2: FVec4::new(self.c0.z, self.c1.z, self.c2.z, self.c3.z),
            c3: FVec4::new(self.c0.w, self.c1.w, self.c2.w, self.c3.w),
        }
    }

    /// Determinant.
    #[inline]
    pub fn determinant(self) -> Fixed32 {
        let a = self.c0.x; let b = self.c1.x; let c = self.c2.x; let d = self.c3.x;
        let e = self.c0.y; let f = self.c1.y; let g = self.c2.y; let h = self.c3.y;
        let i = self.c0.z; let j = self.c1.z; let k = self.c2.z; let l = self.c3.z;
        let m = self.c0.w; let n = self.c1.w; let o = self.c2.w; let p = self.c3.w;

        let kp_lo = k * p - l * o;
        let jp_ln = j * p - l * n;
        let jo_kn = j * o - k * n;
        let ip_lm = i * p - l * m;
        let io_km = i * o - k * m;
        let in_jm = i * n - j * m;

        a * (f * kp_lo - g * jp_ln + h * jo_kn)
            - b * (e * kp_lo - g * ip_lm + h * io_km)
            + c * (e * jp_ln - f * ip_lm + h * in_jm)
            - d * (e * jo_kn - f * io_km + g * in_jm)
    }

    /// Inverse. Returns IDENTITY if singular.
    #[inline]
    pub fn inverse(self) -> Self {
        let det = self.determinant();
        if det == Fixed32::ZERO {
            return Self::IDENTITY;
        }

        let a = self.c0.x; let b = self.c1.x; let c = self.c2.x; let d = self.c3.x;
        let e = self.c0.y; let f = self.c1.y; let g = self.c2.y; let h = self.c3.y;
        let i = self.c0.z; let j = self.c1.z; let k = self.c2.z; let l = self.c3.z;
        let m = self.c0.w; let n = self.c1.w; let o = self.c2.w; let p = self.c3.w;

        let kp_lo = k * p - l * o;
        let jp_ln = j * p - l * n;
        let jo_kn = j * o - k * n;
        let ip_lm = i * p - l * m;
        let io_km = i * o - k * m;
        let in_jm = i * n - j * m;

        let gp_ho = g * p - h * o;
        let fp_hn = f * p - h * n;
        let fo_gn = f * o - g * n;
        let ep_hm = e * p - h * m;
        let eo_gm = e * o - g * m;
        let en_fm = e * n - f * m;

        let gl_hk = g * l - h * k;
        let fl_hj = f * l - h * j;
        let fk_gj = f * k - g * j;
        let el_hi = e * l - h * i;
        let ek_gi = e * k - g * i;
        let ej_fi = e * j - f * i;

        let inv = Fixed32::ONE / det;

        Self {
            c0: FVec4::new(
                (f * kp_lo - g * jp_ln + h * jo_kn) * inv,
                (-(e * kp_lo - g * ip_lm + h * io_km)) * inv,
                (e * jp_ln - f * ip_lm + h * in_jm) * inv,
                (-(e * jo_kn - f * io_km + g * in_jm)) * inv,
            ),
            c1: FVec4::new(
                (-(b * kp_lo - c * jp_ln + d * jo_kn)) * inv,
                (a * kp_lo - c * ip_lm + d * io_km) * inv,
                (-(a * jp_ln - b * ip_lm + d * in_jm)) * inv,
                (a * jo_kn - b * io_km + c * in_jm) * inv,
            ),
            c2: FVec4::new(
                (b * gp_ho - c * fp_hn + d * fo_gn) * inv,
                (-(a * gp_ho - c * ep_hm + d * eo_gm)) * inv,
                (a * fp_hn - b * ep_hm + d * en_fm) * inv,
                (-(a * fo_gn - b * eo_gm + c * en_fm)) * inv,
            ),
            c3: FVec4::new(
                (-(b * gl_hk - c * fl_hj + d * fk_gj)) * inv,
                (a * gl_hk - c * el_hi + d * ek_gi) * inv,
                (-(a * fl_hj - b * el_hi + d * ej_fi)) * inv,
                (a * fk_gj - b * ek_gi + c * ej_fi) * inv,
            ),
        }
    }

    /// Create a rotation matrix from axis and angle.
    #[inline]
    pub fn from_axis_angle(axis: FVec3, angle: Fixed32) -> Self {
        let c = angle.cos_approx();
        let s = angle.sin_approx();
        let t = Fixed32::ONE - c;
        let ax = axis.normalize();
        Self {
            c0: FVec4::new(
                t * ax.x * ax.x + c,
                t * ax.x * ax.y + s * ax.z,
                t * ax.x * ax.z - s * ax.y,
                Fixed32::ZERO,
            ),
            c1: FVec4::new(
                t * ax.y * ax.x - s * ax.z,
                t * ax.y * ax.y + c,
                t * ax.y * ax.z + s * ax.x,
                Fixed32::ZERO,
            ),
            c2: FVec4::new(
                t * ax.z * ax.x + s * ax.y,
                t * ax.z * ax.y - s * ax.x,
                t * ax.z * ax.z + c,
                Fixed32::ZERO,
            ),
            c3: FVec4::new(Fixed32::ZERO, Fixed32::ZERO, Fixed32::ZERO, Fixed32::ONE),
        }
    }

    /// Look-at view matrix (right-handed).
    #[inline]
    pub fn look_at(eye: FVec3, target: FVec3, up: FVec3) -> Self {
        let f = (target - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self {
            c0: FVec4::new(s.x, u.x, -f.x, Fixed32::ZERO),
            c1: FVec4::new(s.y, u.y, -f.y, Fixed32::ZERO),
            c2: FVec4::new(s.z, u.z, -f.z, Fixed32::ZERO),
            c3: FVec4::new(
                -(s.dot(eye)),
                -(u.dot(eye)),
                f.dot(eye),
                Fixed32::ONE,
            ),
        }
    }

    /// Perspective projection matrix (right-handed, infinite far).
    #[inline]
    pub fn perspective(fov_y: Fixed32, aspect: Fixed32, near: Fixed32) -> Self {
        let f = (fov_y / Fixed32::from_f32(2.0)).tan_approx();
        let inv_f = Fixed32::ONE / f;
        Self {
            c0: FVec4::new(inv_f / aspect, Fixed32::ZERO, Fixed32::ZERO, Fixed32::ZERO),
            c1: FVec4::new(Fixed32::ZERO, inv_f, Fixed32::ZERO, Fixed32::ZERO),
            c2: FVec4::new(Fixed32::ZERO, Fixed32::ZERO, Fixed32::from_f32(-1.0), Fixed32::from_f32(-1.0)),
            c3: FVec4::new(Fixed32::ZERO, Fixed32::ZERO, -(near * Fixed32::from_f32(2.0)), Fixed32::ZERO),
        }
    }
}

// SAFETY: M64 is repr(C) with four FVec4 fields, each Pod+Zeroable.
unsafe impl bytemuck::Zeroable for M64 {}
unsafe impl bytemuck::Pod for M64 {}

impl Mul for M64 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self { self.mul_m(other) }
}

// ===========================================================================
// Mat3 (f32)
// ===========================================================================

/// 3x3 column-major f32 matrix (rendering path, normal computation).
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Mat3 {
    pub c0: Vec3,
    pub c1: Vec3,
    pub c2: Vec3,
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        c0: Vec3 { x: 1.0, y: 0.0, z: 0.0 },
        c1: Vec3 { x: 0.0, y: 1.0, z: 0.0 },
        c2: Vec3 { x: 0.0, y: 0.0, z: 1.0 },
    };

    pub const ZERO: Self = Self {
        c0: Vec3::ZERO, c1: Vec3::ZERO, c2: Vec3::ZERO,
    };

    #[inline]
    pub const fn from_columns(c0: Vec3, c1: Vec3, c2: Vec3) -> Self { Self { c0, c1, c2 } }

    #[inline]
    pub fn get(self, col: usize, row: usize) -> f32 {
        let c = match col { 0 => self.c0, 1 => self.c1, 2 => self.c2, _ => panic!("col") };
        match row { 0 => c.x, 1 => c.y, 2 => c.z, _ => panic!("row") }
    }

    fn set(&mut self, col: usize, row: usize, val: f32) {
        let c = match col { 0 => &mut self.c0, 1 => &mut self.c1, 2 => &mut self.c2, _ => panic!("col") };
        match row { 0 => c.x = val, 1 => c.y = val, 2 => c.z = val, _ => panic!("row") }
    }

    #[inline]
    pub fn mul_m(self, other: Self) -> Self {
        let mut r = Self::ZERO;
        for col in 0..3 {
            for row in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += self.get(k, row) * other.get(col, k);
                }
                r.set(col, row, sum);
            }
        }
        r
    }

    #[inline]
    pub fn mul_v(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.c0.x * v.x + self.c1.x * v.y + self.c2.x * v.z,
            y: self.c0.y * v.x + self.c1.y * v.y + self.c2.y * v.z,
            z: self.c0.z * v.x + self.c1.z * v.y + self.c2.z * v.z,
        }
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            c0: Vec3::new(self.c0.x, self.c1.x, self.c2.x),
            c1: Vec3::new(self.c0.y, self.c1.y, self.c2.y),
            c2: Vec3::new(self.c0.z, self.c1.z, self.c2.z),
        }
    }

    #[inline]
    pub fn determinant(self) -> f32 {
        let a = self.c0.x; let b = self.c1.x; let c = self.c2.x;
        let d = self.c0.y; let e = self.c1.y; let f = self.c2.y;
        let g = self.c0.z; let h = self.c1.z; let i = self.c2.z;
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let det = self.determinant();
        if det == 0.0 { return Self::IDENTITY; }
        let inv_det = 1.0 / det;
        let a = self.c0.x; let b = self.c1.x; let c = self.c2.x;
        let d = self.c0.y; let e = self.c1.y; let f = self.c2.y;
        let g = self.c0.z; let h = self.c1.z; let i = self.c2.z;
        Self {
            c0: Vec3::new((e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det),
            c1: Vec3::new((f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det),
            c2: Vec3::new((d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det),
        }
    }
}

// SAFETY: Mat3 is repr(C) with three Vec3 fields.
unsafe impl bytemuck::Zeroable for Mat3 {}
unsafe impl bytemuck::Pod for Mat3 {}

impl Mul for Mat3 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self { self.mul_m(other) }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 { self.mul_v(v) }
}

// ===========================================================================
// Mat4 (f32)
// ===========================================================================

/// 4x4 column-major f32 matrix (rendering path).
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    pub c0: Vec4,
    pub c1: Vec4,
    pub c2: Vec4,
    pub c3: Vec4,
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        c0: Vec4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
        c1: Vec4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 },
        c2: Vec4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 },
        c3: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
    };

    pub const ZERO: Self = Self {
        c0: Vec4::ZERO, c1: Vec4::ZERO, c2: Vec4::ZERO, c3: Vec4::ZERO,
    };

    #[inline]
    pub const fn from_columns(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Self { Self { c0, c1, c2, c3 } }

    #[inline]
    pub fn from_col_array(arr: &[f32; 16]) -> Self {
        Self {
            c0: Vec4::new(arr[0], arr[1], arr[2], arr[3]),
            c1: Vec4::new(arr[4], arr[5], arr[6], arr[7]),
            c2: Vec4::new(arr[8], arr[9], arr[10], arr[11]),
            c3: Vec4::new(arr[12], arr[13], arr[14], arr[15]),
        }
    }

    #[inline]
    pub fn get(self, col: usize, row: usize) -> f32 {
        let c = match col { 0 => self.c0, 1 => self.c1, 2 => self.c2, 3 => self.c3, _ => panic!("col") };
        match row { 0 => c.x, 1 => c.y, 2 => c.z, 3 => c.w, _ => panic!("row") }
    }

    fn set(&mut self, col: usize, row: usize, val: f32) {
        let c = match col { 0 => &mut self.c0, 1 => &mut self.c1, 2 => &mut self.c2, 3 => &mut self.c3, _ => panic!("col") };
        match row { 0 => c.x = val, 1 => c.y = val, 2 => c.z = val, 3 => c.w = val, _ => panic!("row") }
    }

    #[inline]
    pub fn mul_m(self, other: Self) -> Self {
        let mut r = Self::ZERO;
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += self.get(k, row) * other.get(col, k);
                }
                r.set(col, row, sum);
            }
        }
        r
    }

    #[inline]
    pub fn mul_v(self, v: Vec4) -> Vec4 {
        Vec4 {
            x: self.c0.x * v.x + self.c1.x * v.y + self.c2.x * v.z + self.c3.x * v.w,
            y: self.c0.y * v.x + self.c1.y * v.y + self.c2.y * v.z + self.c3.y * v.w,
            z: self.c0.z * v.x + self.c1.z * v.y + self.c2.z * v.z + self.c3.z * v.w,
            w: self.c0.w * v.x + self.c1.w * v.y + self.c2.w * v.z + self.c3.w * v.w,
        }
    }

    #[inline]
    pub fn mul_v3(self, v: Vec3) -> Vec3 {
        let r = self.mul_v(Vec4::new(v.x, v.y, v.z, 1.0));
        Vec3::new(r.x, r.y, r.z)
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            c0: Vec4::new(self.c0.x, self.c1.x, self.c2.x, self.c3.x),
            c1: Vec4::new(self.c0.y, self.c1.y, self.c2.y, self.c3.y),
            c2: Vec4::new(self.c0.z, self.c1.z, self.c2.z, self.c3.z),
            c3: Vec4::new(self.c0.w, self.c1.w, self.c2.w, self.c3.w),
        }
    }

    #[inline]
    pub fn determinant(self) -> f32 {
        let a=self.c0.x;let b=self.c1.x;let c=self.c2.x;let d=self.c3.x;
        let e=self.c0.y;let f=self.c1.y;let g=self.c2.y;let h=self.c3.y;
        let i=self.c0.z;let j=self.c1.z;let k=self.c2.z;let l=self.c3.z;
        let m=self.c0.w;let n=self.c1.w;let o=self.c2.w;let p=self.c3.w;
        let kp_lo=k*p-l*o;let jp_ln=j*p-l*n;let jo_kn=j*o-k*n;
        let ip_lm=i*p-l*m;let io_km=i*o-k*m;let in_jm=i*n-j*m;
        a*(f*kp_lo-g*jp_ln+h*jo_kn)-b*(e*kp_lo-g*ip_lm+h*io_km)
            +c*(e*jp_ln-f*ip_lm+h*in_jm)-d*(e*jo_kn-f*io_km+g*in_jm)
    }

    /// Inverse. Returns IDENTITY if singular.
    #[inline]
    pub fn inverse(self) -> Self {
        let det = self.determinant();
        if det == 0.0 { return Self::IDENTITY; }
        let a0=self.c0.x;let b0=self.c1.x;let c0=self.c2.x;let d0=self.c3.x;
        let e0=self.c0.y;let f0=self.c1.y;let g0=self.c2.y;let h0=self.c3.y;
        let i0=self.c0.z;let j0=self.c1.z;let k0=self.c2.z;let l0=self.c3.z;
        let m0=self.c0.w;let n0=self.c1.w;let o0=self.c2.w;let p0=self.c3.w;
        let inv=1.0/det;
        let kp_lo=k0*p0-l0*o0;let jp_ln=j0*p0-l0*n0;let jo_kn=j0*o0-k0*n0;
        let ip_lm=i0*p0-l0*m0;let io_km=i0*o0-k0*m0;let in_jm=i0*n0-j0*m0;
        let gp_ho=g0*p0-h0*o0;let fp_hn=f0*p0-h0*n0;let fo_gn=f0*o0-g0*n0;
        let ep_hm=e0*p0-h0*m0;let eo_gm=e0*o0-g0*m0;let en_fm=e0*n0-f0*m0;
        let gl_hk=g0*l0-h0*k0;let fl_hj=f0*l0-h0*j0;let fk_gj=f0*k0-g0*j0;
        let el_hi=e0*l0-h0*i0;let ek_gi=e0*k0-g0*i0;let ej_fi=e0*j0-f0*i0;
        Self {
            c0: Vec4::new((f0*kp_lo-g0*jp_ln+h0*jo_kn)*inv,(-(e0*kp_lo-g0*ip_lm+h0*io_km))*inv,(e0*jp_ln-f0*ip_lm+h0*in_jm)*inv,(-(e0*jo_kn-f0*io_km+g0*in_jm))*inv),
            c1: Vec4::new((-(b0*kp_lo-c0*jp_ln+d0*jo_kn))*inv,(a0*kp_lo-c0*ip_lm+d0*io_km)*inv,(-(a0*jp_ln-b0*ip_lm+d0*in_jm))*inv,(a0*jo_kn-b0*io_km+c0*in_jm)*inv),
            c2: Vec4::new((b0*gp_ho-c0*fp_hn+d0*fo_gn)*inv,(-(a0*gp_ho-c0*ep_hm+d0*eo_gm))*inv,(a0*fp_hn-b0*ep_hm+d0*en_fm)*inv,(-(a0*fo_gn-b0*eo_gm+c0*en_fm))*inv),
            c3: Vec4::new((-(b0*gl_hk-c0*fl_hj+d0*fk_gj))*inv,(a0*gl_hk-c0*el_hi+d0*ek_gi)*inv,(-(a0*fl_hj-b0*el_hi+d0*ej_fi))*inv,(a0*fk_gj-b0*ek_gi+c0*ej_fi)*inv),
        }
    }

    /// Look-at view matrix (right-handed).
    #[inline]
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Self {
            c0: Vec4::new(s.x, u.x, -f.x, 0.0),
            c1: Vec4::new(s.y, u.y, -f.y, 0.0),
            c2: Vec4::new(s.z, u.z, -f.z, 0.0),
            c3: Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
        }
    }

    /// Perspective projection matrix (right-handed, infinite far).
    #[inline]
    pub fn perspective(fov_y: f32, aspect: f32, near: f32) -> Self {
        let f = 1.0 / (fov_y * 0.5).tan();
        Self {
            c0: Vec4::new(f / aspect, 0.0, 0.0, 0.0),
            c1: Vec4::new(0.0, f, 0.0, 0.0),
            c2: Vec4::new(0.0, 0.0, -1.0, -1.0),
            c3: Vec4::new(0.0, 0.0, -2.0 * near, 0.0),
        }
    }
}

// SAFETY: Mat4 is repr(C) with four Vec4 fields.
unsafe impl bytemuck::Zeroable for Mat4 {}
unsafe impl bytemuck::Pod for Mat4 {}

impl Mul for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self { self.mul_m(other) }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, v: Vec4) -> Vec4 { self.mul_v(v) }
}
