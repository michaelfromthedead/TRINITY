// Trigonometric lookup table with linear interpolation.
//
// TrigLUT provides fast sin/cos via a precomputed 4096-entry table,
// avoiding the non-determinism of f32 sin/cos across platforms.

use std::sync::OnceLock;

/// Precomputed trigonometric lookup table (4096 entries covering [0, TAU)).
pub struct TrigLUT;

fn compute_sin_table() -> [f32; 4096] {
    let mut table = [0.0f32; 4096];
    let inv_size = 1.0 / 4096.0;
    for i in 0..4096 {
        let x = i as f64 * inv_size as f64 * core::f64::consts::TAU;
        table[i] = x.sin() as f32;
    }
    table
}

fn sin_table() -> &'static [f32; 4096] {
    static SIN_TABLE: OnceLock<[f32; 4096]> = OnceLock::new();
    SIN_TABLE.get_or_init(compute_sin_table)
}

impl TrigLUT {
    /// Number of entries in the lookup table.
    pub const TABLE_SIZE: usize = 4096;

    /// 2 * pi as a 32-bit float.
    pub const TAU: f32 = 6.283_185_5;

    /// Look up sin(x) using linear interpolation.
    ///
    /// Wraps x into [0, TAU) before lookup.
    #[inline]
    pub fn sin(x: f32) -> f32 {
        let x = x % Self::TAU;
        let x = if x < 0.0 { x + Self::TAU } else { x };

        let idx = x * (Self::TABLE_SIZE as f32 / Self::TAU);
        let i = idx as usize;
        let frac = idx - i as f32;

        let i0 = i % Self::TABLE_SIZE;
        let i1 = (i0 + 1) % Self::TABLE_SIZE;

        let table = sin_table();
        let v0 = table[i0];
        let v1 = table[i1];

        v0 + (v1 - v0) * frac
    }

    /// Look up cos(x) using sin(x + pi/2).
    #[inline]
    pub fn cos(x: f32) -> f32 {
        Self::sin(x + core::f32::consts::FRAC_PI_2)
    }

    /// Look up tan(x) = sin(x) / cos(x).
    #[inline]
    pub fn tan(x: f32) -> f32 {
        let c = Self::cos(x);
        if c.abs() < 1e-6 {
            return f32::INFINITY;
        }
        Self::sin(x) / c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_known_values() {
        // sin(0) = 0
        let s0 = TrigLUT::sin(0.0);
        assert!((s0 - 0.0).abs() < 0.001, "sin(0) = {}", s0);

        // sin(pi/2) = 1
        let s1 = TrigLUT::sin(core::f32::consts::FRAC_PI_2);
        assert!((s1 - 1.0).abs() < 0.002, "sin(pi/2) = {}", s1);

        // sin(pi) = 0
        let s2 = TrigLUT::sin(core::f32::consts::PI);
        assert!((s2 - 0.0).abs() < 0.002, "sin(pi) = {}", s2);

        // sin(3pi/2) = -1
        let s3 = TrigLUT::sin(3.0 * core::f32::consts::FRAC_PI_2);
        assert!((s3 - (-1.0)).abs() < 0.002, "sin(3pi/2) = {}", s3);
    }

    #[test]
    fn cos_known_values() {
        // cos(0) = 1
        let c0 = TrigLUT::cos(0.0);
        assert!((c0 - 1.0).abs() < 0.001, "cos(0) = {}", c0);

        // cos(pi/2) = 0
        let c1 = TrigLUT::cos(core::f32::consts::FRAC_PI_2);
        assert!((c1 - 0.0).abs() < 0.002, "cos(pi/2) = {}", c1);

        // cos(pi) = -1
        let c2 = TrigLUT::cos(core::f32::consts::PI);
        assert!((c2 - (-1.0)).abs() < 0.002, "cos(pi) = {}", c2);
    }

    #[test]
    fn sin_wraparound() {
        // sin(x + 2pi) == sin(x)
        let x = 1.0;
        let s1 = TrigLUT::sin(x);
        let s2 = TrigLUT::sin(x + TrigLUT::TAU);
        assert!((s1 - s2).abs() < 0.001, "wraparound mismatch: {} vs {}", s1, s2);
    }

    #[test]
    fn sin_negative() {
        // sin(-x) == -sin(x)
        let x = 1.5;
        let s_pos = TrigLUT::sin(x);
        let s_neg = TrigLUT::sin(-x);
        assert!((s_pos + s_neg).abs() < 0.002, "negation mismatch: {} vs {}", s_pos, s_neg);
    }

    #[test]
    fn cos_accuracy() {
        // Spot-check against known values
        for &(x, expected) in &[
            (0.0, 1.0),
            (1.0, 0.540_302_3),
            (2.0, -0.416_146_8),
            (3.0, -0.989_992_5),
        ] {
            let actual = TrigLUT::cos(x);
            assert!(
                (actual - expected).abs() < 0.002,
                "cos({}) = {} (expected {})",
                x, actual, expected
            );
        }
    }

    #[test]
    fn sin_vs_cos_phase() {
        // sin(x + pi/2) == cos(x)
        let x = 2.5;
        let s_shifted = TrigLUT::sin(x + core::f32::consts::FRAC_PI_2);
        let c = TrigLUT::cos(x);
        assert!((s_shifted - c).abs() < 0.002, "phase mismatch: {} vs {}", s_shifted, c);
    }
}
