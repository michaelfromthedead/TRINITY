// Spatial types for rendering and collision.
//
// AABB    -- Axis-aligned bounding box (24 bytes)
// Ray     -- Ray with origin and direction (24 bytes)
// Frustum -- View frustum defined by 6 planes (96 bytes)
//
// All types derive bytemuck Pod/Zeroable for GPU upload.

use crate::mat::Mat4;
use crate::vec::{Vec3, Vec4};

// ---------------------------------------------------------------------------
// AABB
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box, 24 bytes.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_center_halfext(center: Vec3, halfext: Vec3) -> Self {
        Self {
            min: Vec3::new(center.x - halfext.x, center.y - halfext.y, center.z - halfext.z),
            max: Vec3::new(center.x + halfext.x, center.y + halfext.y, center.z + halfext.z),
        }
    }

    /// Compute the union of two AABBs (smallest AABB containing both).
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    /// Compute the intersection of two AABBs (largest AABB contained in both).
    #[inline]
    pub fn intersection(self, other: Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.max(other.min.x),
                self.min.y.max(other.min.y),
                self.min.z.max(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.min(other.max.x),
                self.max.y.min(other.max.y),
                self.max.z.min(other.max.z),
            ),
        }
    }

    #[inline]
    pub fn contains_point(self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    #[inline]
    pub fn contains_aabb(self, other: Self) -> bool {
        self.contains_point(other.min) && self.contains_point(other.max)
    }

    #[inline]
    pub fn intersects_sphere(self, center: Vec3, radius: f32) -> bool {
        let closest = Vec3::new(
            center.x.clamp(self.min.x, self.max.x),
            center.y.clamp(self.min.y, self.max.y),
            center.z.clamp(self.min.z, self.max.z),
        );
        center.distance(closest) <= radius
    }

    /// Transform the AABB by a 4x4 matrix, producing an axis-aligned result.
    #[inline]
    pub fn transform(self, matrix: Mat4) -> Self {
        let corners = [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ];
        let first = matrix.mul_v3(corners[0]);
        let mut aabb = Self::new(first, first);
        for &c in &corners[1..] {
            let t = matrix.mul_v3(c);
            aabb = aabb.union(Self::new(t, t));
        }
        aabb
    }

    /// Expand the AABB uniformly on all sides.
    #[inline]
    pub fn grow(self, amount: f32) -> Self {
        Self {
            min: Vec3::new(
                self.min.x - amount,
                self.min.y - amount,
                self.min.z - amount,
            ),
            max: Vec3::new(
                self.max.x + amount,
                self.max.y + amount,
                self.max.z + amount,
            ),
        }
    }
}

// SAFETY: AABB is repr(C) with two Vec3 fields.
unsafe impl bytemuck::Zeroable for AABB {}
unsafe impl bytemuck::Pod for AABB {}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    #[inline]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// Evaluate ray at parameter t: `origin + t * direction`.
    #[inline]
    pub fn at(self, t: f32) -> Vec3 {
        Vec3::new(
            self.origin.x + t * self.direction.x,
            self.origin.y + t * self.direction.y,
            self.origin.z + t * self.direction.z,
        )
    }

    /// Ray-AABB intersection using the slabs method.
    #[inline]
    pub fn intersects_aabb(self, aabb: AABB) -> bool {
        let inv_dir = Vec3::new(
            1.0 / self.direction.x,
            1.0 / self.direction.y,
            1.0 / self.direction.z,
        );
        let (t1, t2) = (
            (aabb.min.x - self.origin.x) * inv_dir.x,
            (aabb.max.x - self.origin.x) * inv_dir.x,
        );
        let mut tmin = t1.min(t2);
        let mut tmax = t1.max(t2);

        let (t1, t2) = (
            (aabb.min.y - self.origin.y) * inv_dir.y,
            (aabb.max.y - self.origin.y) * inv_dir.y,
        );
        tmin = tmin.max(t1.min(t2));
        tmax = tmax.min(t1.max(t2));

        let (t1, t2) = (
            (aabb.min.z - self.origin.z) * inv_dir.z,
            (aabb.max.z - self.origin.z) * inv_dir.z,
        );
        tmin = tmin.max(t1.min(t2));
        tmax = tmax.min(t1.max(t2));

        tmax >= tmin && tmax >= 0.0
    }

    /// Ray-sphere intersection test.
    #[inline]
    pub fn intersects_sphere(self, center: Vec3, radius: f32) -> bool {
        let oc = Vec3::new(
            self.origin.x - center.x,
            self.origin.y - center.y,
            self.origin.z - center.z,
        );
        let a = self.direction.dot(self.direction);
        let b = 2.0 * oc.dot(self.direction);
        let c = oc.dot(oc) - radius * radius;
        let discriminant = b * b - 4.0 * a * c;
        discriminant >= 0.0
    }
}

// SAFETY: Ray is repr(C) with two Vec3 fields.
unsafe impl bytemuck::Zeroable for Ray {}
unsafe impl bytemuck::Pod for Ray {}

// ---------------------------------------------------------------------------
// Frustum
// ---------------------------------------------------------------------------

/// View frustum defined by 6 planes, 96 bytes.
///
/// Each plane is (nx, ny, nz, d) with the normal pointing inward.
/// Plane equation: nx*x + ny*y + nz*z + d = 0.
/// Order: [left, right, bottom, top, near, far].
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix (Gribb-Hartmann).
    #[inline]
    pub fn from_view_proj(matrix: Mat4) -> Self {
        // Rows from column-major storage
        let row0 = Vec4::new(matrix.c0.x, matrix.c1.x, matrix.c2.x, matrix.c3.x);
        let row1 = Vec4::new(matrix.c0.y, matrix.c1.y, matrix.c2.y, matrix.c3.y);
        let row2 = Vec4::new(matrix.c0.z, matrix.c1.z, matrix.c2.z, matrix.c3.z);
        let row3 = Vec4::new(matrix.c0.w, matrix.c1.w, matrix.c2.w, matrix.c3.w);

        let raw_planes = [
            row3 + row0, // left
            row3 - row0, // right
            row3 + row1, // bottom
            row3 - row1, // top
            row3 + row2, // near
            row3 - row2, // far
        ];

        let mut planes = [Vec4::ZERO; 6];
        for (i, p) in raw_planes.into_iter().enumerate() {
            let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            if len > 0.0 {
                planes[i] = Vec4::new(p.x / len, p.y / len, p.z / len, p.w / len);
            } else {
                planes[i] = p;
            }
        }
        Self { planes }
    }

    /// Test if an AABB intersects the frustum.
    #[inline]
    pub fn contains_aabb(self, aabb: AABB) -> bool {
        for &plane in &self.planes {
            let px = if plane.x >= 0.0 { aabb.max.x } else { aabb.min.x };
            let py = if plane.y >= 0.0 { aabb.max.y } else { aabb.min.y };
            let pz = if plane.z >= 0.0 { aabb.max.z } else { aabb.min.z };
            if plane.x * px + plane.y * py + plane.z * pz + plane.w < 0.0 {
                return false;
            }
        }
        true
    }

    /// Test if a sphere intersects the frustum.
    #[inline]
    pub fn test_sphere(self, center: Vec3, radius: f32) -> bool {
        for &plane in &self.planes {
            let dist = plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
            if dist < -radius {
                return false;
            }
        }
        true
    }

    /// Test if a point is inside the frustum.
    #[inline]
    pub fn test_point(self, point: Vec3) -> bool {
        for &plane in &self.planes {
            if plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

// SAFETY: Frustum is repr(C) with six Vec4 fields.
unsafe impl bytemuck::Zeroable for Frustum {}
unsafe impl bytemuck::Pod for Frustum {}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod spatial_tests {
    use crate::mat::Mat4;
    use crate::spatial::*;
    use crate::vec::Vec3;

    // ---- AABB ----

    #[test]
    fn aabb_new() {
        let a = AABB::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(a.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn aabb_from_center_halfext() {
        let a = AABB::from_center_halfext(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_eq!(a.max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn aabb_contains_point() {
        let a = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(a.contains_point(Vec3::ZERO));
        assert!(a.contains_point(Vec3::new(1.0, 1.0, 1.0)));
        assert!(!a.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn aabb_union() {
        let a = AABB::new(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(0.0, 0.0, 0.0));
        let b = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let u = a.union(b);
        assert_eq!(u.min, Vec3::new(-2.0, -2.0, -2.0));
        assert_eq!(u.max, Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn aabb_intersection() {
        let a = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(2.0, 2.0, 2.0));
        let b = AABB::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 3.0, 3.0));
        let i = a.intersection(b);
        assert_eq!(i.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(i.max, Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn aabb_contains_aabb() {
        let outer = AABB::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0));
        let inner = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(outer.contains_aabb(inner));
        assert!(!inner.contains_aabb(outer));
    }

    #[test]
    fn aabb_intersects_sphere() {
        let a = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(a.intersects_sphere(Vec3::ZERO, 1.0));
        assert!(a.intersects_sphere(Vec3::new(2.0, 0.0, 0.0), 1.5));
        assert!(!a.intersects_sphere(Vec3::new(10.0, 0.0, 0.0), 1.0));
    }

    #[test]
    fn aabb_grow() {
        let a = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let g = a.grow(1.0);
        assert_eq!(g.min, Vec3::new(-2.0, -2.0, -2.0));
        assert_eq!(g.max, Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn aabb_transform_identity() {
        let a = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let t = a.transform(Mat4::IDENTITY);
        assert_eq!(t, a);
    }

    #[test]
    fn aabb_bytemuck_pod() {
        let a = AABB::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let bytes: &[u8] = bytemuck::bytes_of(&a);
        assert_eq!(bytes.len(), 24);
    }

    // ---- Ray ----

    #[test]
    fn ray_at() {
        let r = Ray::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(r.at(5.0), Vec3::new(6.0, 2.0, 3.0));
    }

    #[test]
    fn ray_intersects_aabb_hit() {
        let r = Ray::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let a = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(r.intersects_aabb(a));
    }

    #[test]
    fn ray_intersects_aabb_miss() {
        let r = Ray::new(Vec3::new(-10.0, 10.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let a = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(!r.intersects_aabb(a));
    }

    #[test]
    fn ray_intersects_sphere_hit() {
        let r = Ray::new(Vec3::new(-10.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(r.intersects_sphere(Vec3::ZERO, 1.0));
    }

    #[test]
    fn ray_intersects_sphere_miss() {
        let r = Ray::new(Vec3::new(-10.0, 10.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(!r.intersects_sphere(Vec3::ZERO, 1.0));
    }

    #[test]
    fn ray_bytemuck_pod() {
        let r = Ray::new(Vec3::ZERO, Vec3::UNIT_X);
        let bytes: &[u8] = bytemuck::bytes_of(&r);
        assert_eq!(bytes.len(), 24);
    }

    // ---- Frustum ----

    #[test]
    fn frustum_from_view_proj_identity() {
        let f = Frustum::from_view_proj(Mat4::IDENTITY);
        assert_eq!(f.planes.len(), 6);
    }

    #[test]
    fn frustum_test_point_inside() {
        let f = Frustum::from_view_proj(Mat4::IDENTITY);
        assert!(f.test_point(Vec3::new(0.5, 0.5, 0.5)));
    }

    #[test]
    fn frustum_contains_aabb() {
        let f = Frustum::from_view_proj(Mat4::IDENTITY);
        let small = AABB::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5));
        assert!(f.contains_aabb(small));
    }

    #[test]
    fn frustum_test_sphere() {
        let f = Frustum::from_view_proj(Mat4::IDENTITY);
        assert!(f.test_sphere(Vec3::new(0.5, 0.5, 0.5), 0.1));
    }

    #[test]
    fn frustum_bytemuck_pod() {
        let f = Frustum::from_view_proj(Mat4::IDENTITY);
        let bytes: &[u8] = bytemuck::bytes_of(&f);
        assert_eq!(bytes.len(), 96);
    }
}
