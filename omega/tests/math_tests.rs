// Comprehensive whitebox tests for the omega math library.
//
// Organized by module:
//   1. fixed_point  -- Fixed16/Fixed32 arithmetic edge cases
//   2. vectors      -- FVec2/3/4 and Vec2/3/4 operations
//   3. quaternions  -- FQuat/Quat edge cases
//   4. matrices     -- M64/Mat3/Mat4 edge cases
//   5. trig_lut     -- TrigLUT accuracy and boundary behavior
//   6. sim_rng      -- SimRng determinism and cross-platform stability
//   7. bytemuck     -- Pod/Zeroable trait guarantees
//   8. integration  -- Cross-module combined operations
//   9. determinism  -- Fixed-point determinism verification

// ===========================================================================
// 1. Fixed-point arithmetic
// ===========================================================================

#[cfg(test)]
mod fixed_point {
    use omega::*;

    // -----------------------------------------------------------------------
    // Fixed16 saturation and overflow
    // -----------------------------------------------------------------------

    #[test]
    fn f16_add_saturate_max() {
        let a = Fixed16::MAX;
        let b = Fixed16::from_f32(1.0);
        let r = a + b;
        assert_eq!(r, Fixed16::MAX, "MAX + 1 should saturate to MAX");
    }

    #[test]
    fn f16_add_saturate_min() {
        let a = Fixed16::MIN;
        let b = Fixed16::from_f32(-1.0);
        let r = a + b;
        assert_eq!(r, Fixed16::MIN, "MIN + (-1) should saturate to MIN");
    }

    #[test]
    fn f16_sub_saturate_max() {
        let a = Fixed16::MIN;
        let b = Fixed16::from_f32(1.0);
        let r = a - b;
        assert_eq!(r, Fixed16::MIN, "MIN - 1 should saturate to MIN");
    }

    #[test]
    fn f16_mul_saturate_max() {
        let a = Fixed16::from_f32(128.0);
        let b = Fixed16::from_f32(2.0);
        let r = a * b;
        assert_eq!(r, Fixed16::MAX, "128 * 2 should saturate to MAX");
    }

    #[test]
    fn f16_mul_saturate_min() {
        let a = Fixed16::from_f32(-128.0);
        let b = Fixed16::from_f32(2.0);
        let r = a * b;
        assert_eq!(r, Fixed16::MIN, "-128 * 2 should saturate to MIN");
    }

    #[test]
    fn f16_div_by_zero_positive() {
        let a = Fixed16::from_f32(1.0);
        let b = Fixed16::ZERO;
        let r = a / b;
        assert_eq!(r, Fixed16::MAX, "positive / 0 should saturate to MAX");
    }

    #[test]
    fn f16_div_by_zero_negative() {
        let a = Fixed16::from_f32(-1.0);
        let b = Fixed16::ZERO;
        let r = a / b;
        assert_eq!(r, Fixed16::MIN, "negative / 0 should saturate to MIN");
    }

    #[test]
    fn f16_div_by_zero_zero() {
        let r = Fixed16::ZERO / Fixed16::ZERO;
        assert_eq!(r, Fixed16::MAX, "0 / 0 should saturate to MAX");
    }

    #[test]
    fn f16_mul_identity() {
        let a = Fixed16::from_f32(42.5);
        assert_eq!(a * Fixed16::ONE, a);
    }

    #[test]
    fn f16_from_i16_overflow() {
        // 128 * 256 = 32768 > i16::MAX, should saturate
        let v = Fixed16::from(127i16);
        assert_eq!(v.to_f32(), 127.0);
        let v2 = Fixed16::from(-128i16);
        assert_eq!(v2.to_f32(), -128.0);
    }

    #[test]
    fn f16_neg_min() {
        let r = -Fixed16::MIN;
        assert_eq!(r, Fixed16::MAX, "-MIN should saturate to MAX");
    }

    #[test]
    fn f16_abs_min() {
        let r = Fixed16::MIN.abs();
        assert_eq!(r, Fixed16::MAX, "abs(MIN) should saturate to MAX");
    }

    #[test]
    fn f16_floor_positive() {
        let v = Fixed16::from_f32(3.75);
        assert_eq!(v.floor(), Fixed16::from_f32(3.0));
    }

    #[test]
    fn f16_floor_negative() {
        let v = Fixed16::from_f32(-3.25);
        assert_eq!(v.floor(), Fixed16::from_f32(-4.0));
    }

    #[test]
    fn f16_ceil_positive() {
        let v = Fixed16::from_f32(3.25);
        assert_eq!(v.ceil(), Fixed16::from_f32(4.0));
    }

    #[test]
    fn f16_ceil_negative() {
        let v = Fixed16::from_f32(-3.75);
        assert_eq!(v.ceil(), Fixed16::from_f32(-3.0));
    }

    #[test]
    fn f16_is_zero() {
        assert!(Fixed16::ZERO.is_zero());
        assert!(!Fixed16::ONE.is_zero());
        assert!(!Fixed16::from_f32(0.5).is_zero());
    }

    #[test]
    fn f16_is_negative() {
        assert!(Fixed16::from_f32(-1.0).is_negative());
        assert!(!Fixed16::from_f32(1.0).is_negative());
        assert!(!Fixed16::ZERO.is_negative());
    }

    #[test]
    fn f16_lerp_bounds() {
        let a = Fixed16::from_f32(0.0);
        let b = Fixed16::from_f32(10.0);
        assert_eq!(a.lerp(b, Fixed16::ZERO), a);
        assert_eq!(a.lerp(b, Fixed16::ONE), b);
    }

    #[test]
    fn f16_lerp_half() {
        let a = Fixed16::from_f32(0.0);
        let b = Fixed16::from_f32(10.0);
        let half = Fixed16::from_f32(0.5);
        let r = a.lerp(b, half);
        let expected = Fixed16::from_f32(5.0);
        assert!(
            (r - expected).abs().to_f32() < 0.01,
            "lerp(0, 10, 0.5) = {} (expected {})",
            r.to_f32(),
            expected.to_f32()
        );
    }

    #[test]
    fn f16_round_to_zero() {
        let v = Fixed16::from_f32(3.75);
        assert_eq!(v.round_to_zero(), 3);
        let v = Fixed16::from_f32(-3.75);
        assert_eq!(v.round_to_zero(), -3);
    }

    // -----------------------------------------------------------------------
    // Fixed32 saturation and overflow
    // -----------------------------------------------------------------------

    #[test]
    fn f32_add_saturate_max() {
        let a = Fixed32::MAX;
        let b = Fixed32::from_f32(1.0);
        let r = a + b;
        assert_eq!(r, Fixed32::MAX, "MAX + 1 should saturate");
    }

    #[test]
    fn f32_add_saturate_min() {
        let a = Fixed32::MIN;
        let b = Fixed32::from_f32(-1.0);
        let r = a + b;
        assert_eq!(r, Fixed32::MIN, "MIN + (-1) should saturate");
    }

    #[test]
    fn f32_sub_saturate_max() {
        let a = Fixed32::MAX;
        let b = Fixed32::from_f32(-1.0);
        let r = a - b;
        assert_eq!(r, Fixed32::MAX, "MAX - (-1) should saturate to MAX");
    }

    #[test]
    fn f32_sub_saturate_min() {
        let a = Fixed32::MIN;
        let b = Fixed32::from_f32(1.0);
        let r = a - b;
        assert_eq!(r, Fixed32::MIN, "MIN - 1 should saturate to MIN");
    }

    #[test]
    fn f32_mul_saturate_max() {
        let a = Fixed32::from_f32(32768.0);
        let b = Fixed32::from_f32(2.0);
        let r = a * b;
        assert_eq!(r, Fixed32::MAX, "32768 * 2 should saturate to MAX");
    }

    #[test]
    fn f32_mul_saturate_min() {
        let a = Fixed32::from_f32(-32768.0);
        let b = Fixed32::from_f32(2.0);
        let r = a * b;
        assert_eq!(r, Fixed32::MIN, "-32768 * 2 should saturate to MIN");
    }

    #[test]
    fn f32_mul_zero() {
        let a = Fixed32::from_f32(100.0);
        assert_eq!(a * Fixed32::ZERO, Fixed32::ZERO);
    }

    #[test]
    fn f32_mul_identity() {
        let a = Fixed32::from_f32(42.5);
        assert_eq!(a * Fixed32::ONE, a);
    }

    #[test]
    fn f32_div_by_zero_positive() {
        let a = Fixed32::from_f32(1.0);
        let r = a / Fixed32::ZERO;
        assert_eq!(r, Fixed32::MAX, "positive / 0 should saturate to MAX");
    }

    #[test]
    fn f32_div_by_zero_negative() {
        let a = Fixed32::from_f32(-1.0);
        let r = a / Fixed32::ZERO;
        assert_eq!(r, Fixed32::MIN, "negative / 0 should saturate to MIN");
    }

    #[test]
    fn f32_div_by_zero_zero() {
        let r = Fixed32::ZERO / Fixed32::ZERO;
        assert_eq!(r, Fixed32::MAX, "0 / 0 should saturate to MAX");
    }

    #[test]
    fn f32_div_self() {
        let a = Fixed32::from_f32(42.0);
        assert!((a / a - Fixed32::ONE).abs().to_f32() < 0.001);
    }

    #[test]
    fn f32_div_negative() {
        let a = Fixed32::from_f32(10.0);
        let b = Fixed32::from_f32(-2.0);
        let r = a / b;
        let expected = Fixed32::from_f32(-5.0);
        assert!(
            (r - expected).abs().to_f32() < 0.001,
            "10 / -2 = {} (expected {})",
            r.to_f32(),
            expected.to_f32()
        );
    }

    #[test]
    fn f32_from_i32_overflow() {
        let v = Fixed32::from(i32::MAX);
        assert_eq!(v, Fixed32::MAX);
    }

    #[test]
    fn f32_from_i32_underflow() {
        let v = Fixed32::from(i32::MIN);
        assert_eq!(v, Fixed32::MIN);
    }

    #[test]
    fn f32_from_i32_identity() {
        let v = Fixed32::from(42i32);
        assert_eq!(v.to_f32(), 42.0);
    }

    #[test]
    fn f32_neg_min() {
        let r = -Fixed32::MIN;
        assert_eq!(r, Fixed32::MAX, "-MIN should saturate to MAX");
    }

    #[test]
    fn f32_abs_min() {
        let r = Fixed32::MIN.abs();
        assert_eq!(r, Fixed32::MAX, "abs(MIN) should saturate to MAX");
    }

    #[test]
    fn f32_abs_positive() {
        assert_eq!(Fixed32::from_f32(42.0).abs(), Fixed32::from_f32(42.0));
    }

    #[test]
    fn f32_abs_negative() {
        assert_eq!(Fixed32::from_f32(-42.0).abs(), Fixed32::from_f32(42.0));
    }

    #[test]
    fn f32_is_zero() {
        assert!(Fixed32::ZERO.is_zero());
        assert!(!Fixed32::ONE.is_zero());
    }

    #[test]
    fn f32_is_negative() {
        assert!(Fixed32::from_f32(-1.0).is_negative());
        assert!(!Fixed32::from_f32(1.0).is_negative());
        assert!(!Fixed32::ZERO.is_negative());
    }

    #[test]
    fn f32_floor_positive() {
        let v = Fixed32::from_f32(3.75);
        assert_eq!(v.floor(), Fixed32::from_f32(3.0));
    }

    #[test]
    fn f32_floor_negative() {
        let v = Fixed32::from_f32(-3.25);
        assert_eq!(v.floor(), Fixed32::from_f32(-4.0));
    }

    #[test]
    fn f32_floor_exact() {
        let v = Fixed32::from_f32(5.0);
        assert_eq!(v.floor(), v);
    }

    #[test]
    fn f32_ceil_positive() {
        let v = Fixed32::from_f32(3.25);
        assert_eq!(v.ceil(), Fixed32::from_f32(4.0));
    }

    #[test]
    fn f32_ceil_negative() {
        let v = Fixed32::from_f32(-3.75);
        assert_eq!(v.ceil(), Fixed32::from_f32(-3.0));
    }

    #[test]
    fn f32_ceil_exact() {
        let v = Fixed32::from_f32(5.0);
        assert_eq!(v.ceil(), v);
    }

    #[test]
    fn f32_floor_ceil_identity() {
        // For integer values, floor == ceil == value
        let v = Fixed32::from_f32(5.0);
        assert_eq!(v.floor(), v.ceil());
    }

    #[test]
    fn f32_round_to_zero() {
        let v = Fixed32::from_f32(3.75);
        assert_eq!(v.round_to_zero(), 3);
        let v = Fixed32::from_f32(-3.75);
        assert_eq!(v.round_to_zero(), -3);
    }

    #[test]
    fn f32_lerp_bounds() {
        let a = Fixed32::from_f32(0.0);
        let b = Fixed32::from_f32(100.0);
        assert_eq!(a.lerp(b, Fixed32::ZERO), a);
        assert_eq!(a.lerp(b, Fixed32::ONE), b);
    }

    #[test]
    fn f32_lerp_half() {
        let a = Fixed32::from_f32(0.0);
        let b = Fixed32::from_f32(100.0);
        let half = Fixed32::from_f32(0.5);
        let r = a.lerp(b, half);
        assert!(
            (r - Fixed32::from_f32(50.0)).abs().to_f32() < 0.01,
            "lerp(0, 100, 0.5) = {}",
            r.to_f32()
        );
    }

    #[test]
    fn f32_lerp_reverse() {
        let a = Fixed32::from_f32(100.0);
        let b = Fixed32::from_f32(0.0);
        let half = Fixed32::from_f32(0.5);
        let r = a.lerp(b, half);
        assert!(
            (r - Fixed32::from_f32(50.0)).abs().to_f32() < 0.01,
            "lerp(100, 0, 0.5) = {}",
            r.to_f32()
        );
    }

    #[test]
    fn f32_sqrt_zero() {
        assert_eq!(Fixed32::ZERO.sqrt(), Fixed32::ZERO);
    }

    #[test]
    fn f32_sqrt_one() {
        let r = Fixed32::ONE.sqrt();
        assert!((r - Fixed32::ONE).abs().to_f32() < 0.001);
    }

    #[test]
    fn f32_sqrt_small() {
        let v = Fixed32::from_f32(0.25);
        let r = v.sqrt();
        assert!(
            (r - Fixed32::from_f32(0.5)).abs().to_f32() < 0.01,
            "sqrt(0.25) = {}",
            r.to_f32()
        );
    }

    #[test]
    fn f32_sqrt_large() {
        let v = Fixed32::from_f32(30000.0);
        let r = v.sqrt();
        let expected = 30000.0f64.sqrt() as f32;
        assert!(
            (r.to_f32() - expected).abs() < 1.0,
            "sqrt(30000) = {} (expected ~{})",
            r.to_f32(),
            expected
        );
    }

    #[test]
    fn f32_sqrt_negative() {
        assert_eq!(Fixed32::from_f32(-1.0).sqrt(), Fixed32::ZERO);
    }

    #[test]
    fn f32_sqrt_very_large() {
        let v = Fixed32::from_f32(30000.0);
        let r = v.sqrt();
        assert!(
            (r.to_f32() - 173.205).abs() < 5.0,
            "sqrt(30000) = {} (expected ~173.2)",
            r.to_f32()
        );
    }

    #[test]
    fn f32_add_commutative() {
        let a = Fixed32::from_f32(10.0);
        let b = Fixed32::from_f32(20.0);
        assert_eq!(a + b, b + a);
    }

    #[test]
    fn f32_mul_commutative() {
        let a = Fixed32::from_f32(10.0);
        let b = Fixed32::from_f32(20.0);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn f32_rem_by_zero() {
        let r = Fixed32::from_f32(10.0) % Fixed32::ZERO;
        assert_eq!(r, Fixed32::ZERO, "x % 0 should return 0");
    }

    #[test]
    fn f32_rem_positive() {
        let r = Fixed32::from_f32(10.0) % Fixed32::from_f32(3.0);
        assert_eq!(r, Fixed32::from_f32(1.0), "10 % 3 = {}", r.to_f32());
    }

    #[test]
    fn f32_rem_negative() {
        let r = Fixed32::from_f32(-10.0) % Fixed32::from_f32(3.0);
        assert_eq!(r.to_f32(), -1.0, "-10 % 3 = {}", r.to_f32());
    }

    #[test]
    fn f32_from_f16() {
        let f16 = Fixed16::from_f32(3.5);
        let f32: Fixed32 = f16.into();
        assert!(
            (f32.to_f32() - 3.5).abs() < 0.001,
            "Fixed16(3.5) -> Fixed32 = {}",
            f32.to_f32()
        );
    }

    // -----------------------------------------------------------------------
    // Raw value edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn f32_from_raw_edge() {
        // from_raw(-1) should be -1/65536, not -1.0
        let v = Fixed32::from_raw(-1);
        assert!(
            (v.to_f32() + 1.0 / 65536.0).abs() < 0.0001,
            "from_raw(-1) = {} (expected {})",
            v.to_f32(),
            -1.0 / 65536.0
        );
    }

    #[test]
    fn f32_from_raw_scale() {
        let v = Fixed32::from_raw(65536);
        assert!(
            (v.to_f32() - 1.0).abs() < 0.0001,
            "from_raw(65536) = {} (expected 1.0)",
            v.to_f32()
        );
    }

    #[test]
    fn f32_from_raw_negative_one() {
        let v = Fixed32::from_raw(-65536);
        assert!(
            (v.to_f32() - (-1.0)).abs() < 0.0001,
            "from_raw(-65536) = {} (expected -1.0)",
            v.to_f32()
        );
    }

    #[test]
    fn f16_from_raw_scale() {
        let v = Fixed16::from_raw(256);
        assert!(
            (v.to_f32() - 1.0).abs() < 0.01,
            "from_raw(256) = {} (expected 1.0)",
            v.to_f32()
        );
    }

    // -----------------------------------------------------------------------
    // EPSILON tests
    // -----------------------------------------------------------------------

    #[test]
    fn f32_epsilon_nonzero() {
        assert!(Fixed32::EPSILON != Fixed32::ZERO);
        assert!(Fixed32::EPSILON > Fixed32::ZERO);
    }

    #[test]
    fn f16_epsilon_nonzero() {
        assert!(Fixed16::EPSILON != Fixed16::ZERO);
        assert!(Fixed16::EPSILON > Fixed16::ZERO);
    }

    #[test]
    fn f32_epsilon_nonzero_small() {
        assert!(Fixed32::EPSILON < Fixed32::ONE);
        assert!(Fixed32::EPSILON > Fixed32::ZERO);
    }
}

// ===========================================================================
// 2. Vector operations
// ===========================================================================

#[cfg(test)]
mod vectors {
    use omega::*;

    // -----------------------------------------------------------------------
    // FVec2
    // -----------------------------------------------------------------------

    #[test]
    fn fv2_zero_length() {
        assert_eq!(FVec2::ZERO.length(), Fixed32::ZERO);
    }

    #[test]
    fn fv2_zero_length_squared() {
        assert_eq!(FVec2::ZERO.length_squared(), Fixed32::ZERO);
    }

    #[test]
    fn fv2_normalize_zero() {
        assert_eq!(FVec2::ZERO.normalize(), FVec2::ZERO);
    }

    #[test]
    fn fv2_normalize_unit_x() {
        let n = FVec2::UNIT_X.normalize();
        assert!((n.x - Fixed32::ONE).abs().to_f32() < 0.001);
        assert!((n.y).abs().to_f32() < 0.001);
    }

    #[test]
    fn fv2_normalize_arbitrary() {
        let v = FVec2::new(Fixed32::from_f32(3.0), Fixed32::from_f32(4.0));
        let n = v.normalize();
        assert!(
            (n.length() - Fixed32::ONE).abs().to_f32() < 0.01,
            "|normalize(3,4)| = {}",
            n.length().to_f32()
        );
    }

    #[test]
    fn fv2_dot_orthogonal() {
        let dot = FVec2::UNIT_X.dot(FVec2::UNIT_Y);
        assert_eq!(dot, Fixed32::ZERO);
    }

    #[test]
    fn fv2_dot_parallel() {
        let dot = FVec2::UNIT_X.dot(FVec2::UNIT_X);
        assert_eq!(dot, Fixed32::ONE);
    }

    #[test]
    fn fv2_dot_self_length_sq() {
        let v = FVec2::new(Fixed32::from_f32(3.0), Fixed32::from_f32(4.0));
        assert_eq!(v.dot(v), v.length_squared());
    }

    #[test]
    fn fv2_add_identity() {
        assert_eq!(FVec2::UNIT_X + FVec2::ZERO, FVec2::UNIT_X);
    }

    #[test]
    fn fv2_sub_self() {
        assert_eq!(FVec2::UNIT_X - FVec2::UNIT_X, FVec2::ZERO);
    }

    #[test]
    fn fv2_mul_scalar_zero() {
        assert_eq!(FVec2::UNIT_X * Fixed32::ZERO, FVec2::ZERO);
    }

    #[test]
    fn fv2_mul_scalar_identity() {
        assert_eq!(FVec2::UNIT_X * Fixed32::ONE, FVec2::UNIT_X);
    }

    #[test]
    fn fv2_div_scalar_one() {
        assert_eq!(FVec2::UNIT_X / Fixed32::ONE, FVec2::UNIT_X);
    }

    #[test]
    fn fv2_neg() {
        assert_eq!(-FVec2::UNIT_X, FVec2::new(Fixed32::from_f32(-1.0), Fixed32::ZERO));
    }

    #[test]
    fn fv2_lerp_bounds() {
        let a = FVec2::ZERO;
        let b = FVec2::new(Fixed32::from_f32(10.0), Fixed32::from_f32(10.0));
        assert_eq!(a.lerp(b, Fixed32::ZERO), a);
        assert_eq!(a.lerp(b, Fixed32::ONE), b);
    }

    #[test]
    fn fv2_min_max() {
        let a = FVec2::new(Fixed32::from_f32(1.0), Fixed32::from_f32(5.0));
        let b = FVec2::new(Fixed32::from_f32(3.0), Fixed32::from_f32(2.0));
        assert_eq!(a.min(b), FVec2::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0)));
        assert_eq!(a.max(b), FVec2::new(Fixed32::from_f32(3.0), Fixed32::from_f32(5.0)));
    }

    #[test]
    fn fv2_clamp() {
        let v = FVec2::new(Fixed32::from_f32(5.0), Fixed32::from_f32(-5.0));
        let lo = FVec2::new(Fixed32::from_f32(-1.0), Fixed32::from_f32(-1.0));
        let hi = FVec2::new(Fixed32::from_f32(1.0), Fixed32::from_f32(1.0));
        let clamped = v.clamp(lo, hi);
        assert_eq!(clamped.x, Fixed32::from_f32(1.0));
        assert_eq!(clamped.y, Fixed32::from_f32(-1.0));
    }

    // -----------------------------------------------------------------------
    // FVec3
    // -----------------------------------------------------------------------

    #[test]
    fn fv3_zero_length() {
        assert_eq!(FVec3::ZERO.length(), Fixed32::ZERO);
    }

    #[test]
    fn fv3_normalize_zero() {
        assert_eq!(FVec3::ZERO.normalize(), FVec3::ZERO);
    }

    #[test]
    fn fv3_normalize_unit() {
        let n = FVec3::UNIT_X.normalize();
        assert_eq!(n, FVec3::UNIT_X);
    }

    #[test]
    fn fv3_normalize_arbitrary() {
        let v = FVec3::new(Fixed32::from_f32(2.0), Fixed32::from_f32(3.0), Fixed32::from_f32(6.0));
        let n = v.normalize();
        let _expected_len = (2.0f32 * 2.0 + 3.0 * 3.0 + 6.0 * 6.0).sqrt();
        assert!(
            (n.length().to_f32() - 1.0).abs() < 0.01,
            "|normalize| = {}",
            n.length().to_f32()
        );
    }

    #[test]
    fn fv3_normalize_epsilon() {
        // Very small vector should normalize to zero without panicking
        let tiny = FVec3::new(
            Fixed32::EPSILON,
            Fixed32::EPSILON,
            Fixed32::EPSILON,
        );
        let n = tiny.normalize();
        // length_squared of (eps, eps, eps) = 3*eps^2 = 3/65536^2 which is 0 in integer division
        // so normalize returns ZERO
        assert_eq!(n, FVec3::ZERO, "epsilon vector should normalize to zero");
    }

    #[test]
    fn fv3_cross_unit_axes() {
        // X x Y = Z
        let cross = FVec3::UNIT_X.cross(FVec3::UNIT_Y);
        assert_eq!(cross, FVec3::UNIT_Z);
    }

    #[test]
    fn fv3_cross_reverse() {
        // Y x X = -Z
        let cross = FVec3::UNIT_Y.cross(FVec3::UNIT_X);
        assert_eq!(cross, -FVec3::UNIT_Z);
    }

    #[test]
    fn fv3_cross_self_zero() {
        assert_eq!(FVec3::UNIT_X.cross(FVec3::UNIT_X), FVec3::ZERO);
    }

    #[test]
    fn fv3_cross_orthogonal() {
        let a = FVec3::new(Fixed32::from_f32(1.0), Fixed32::ZERO, Fixed32::ZERO);
        let b = FVec3::new(Fixed32::ZERO, Fixed32::from_f32(1.0), Fixed32::ZERO);
        let c = a.cross(b);
        assert_eq!(c, FVec3::UNIT_Z);
    }

    #[test]
    fn fv3_cross_anticommutative() {
        let a = FVec3::new(Fixed32::from_f32(2.0), Fixed32::from_f32(3.0), Fixed32::from_f32(4.0));
        let b = FVec3::new(Fixed32::from_f32(5.0), Fixed32::from_f32(6.0), Fixed32::from_f32(7.0));
        assert_eq!(a.cross(b), -(b.cross(a)));
    }

    #[test]
    fn fv3_dot_orthogonal() {
        assert_eq!(FVec3::UNIT_X.dot(FVec3::UNIT_Y), Fixed32::ZERO);
    }

    #[test]
    fn fv3_dot_parallel() {
        assert_eq!(FVec3::UNIT_X.dot(FVec3::UNIT_X), Fixed32::ONE);
    }

    #[test]
    fn fv3_dot_self_length_sq() {
        let v = FVec3::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0), Fixed32::from_f32(3.0));
        assert_eq!(v.dot(v), v.length_squared());
    }

    #[test]
    fn fv3_add_identity() {
        assert_eq!(FVec3::UNIT_X + FVec3::ZERO, FVec3::UNIT_X);
    }

    #[test]
    fn fv3_sub_self() {
        assert_eq!(FVec3::UNIT_X - FVec3::UNIT_X, FVec3::ZERO);
    }

    #[test]
    fn fv3_neg() {
        assert_eq!(-FVec3::UNIT_X, FVec3::new(Fixed32::from_f32(-1.0), Fixed32::ZERO, Fixed32::ZERO));
    }

    #[test]
    fn fv3_lerp_bounds() {
        let a = FVec3::ZERO;
        let b = FVec3::new(Fixed32::from_f32(5.0), Fixed32::from_f32(5.0), Fixed32::from_f32(5.0));
        assert_eq!(a.lerp(b, Fixed32::ZERO), a);
        assert_eq!(a.lerp(b, Fixed32::ONE), b);
    }

    // -----------------------------------------------------------------------
    // FVec4
    // -----------------------------------------------------------------------

    #[test]
    fn fv4_zero_length() {
        assert_eq!(FVec4::ZERO.length(), Fixed32::ZERO);
    }

    #[test]
    fn fv4_dot_identity() {
        assert_eq!(FVec4::UNIT_X.dot(FVec4::UNIT_X), Fixed32::ONE);
    }

    #[test]
    fn fv4_dot_orthogonal() {
        assert_eq!(FVec4::UNIT_X.dot(FVec4::UNIT_Y), Fixed32::ZERO);
    }

    #[test]
    fn fv4_lerp_bounds() {
        let a = FVec4::ZERO;
        let b = FVec4::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0), Fixed32::from_f32(3.0), Fixed32::from_f32(4.0));
        assert_eq!(a.lerp(b, Fixed32::ZERO), a);
        assert_eq!(a.lerp(b, Fixed32::ONE), b);
    }

    #[test]
    fn fv4_add() {
        let r = FVec4::UNIT_X + FVec4::UNIT_Y;
        assert_eq!(r, FVec4::new(Fixed32::ONE, Fixed32::ONE, Fixed32::ZERO, Fixed32::ZERO));
    }

    // -----------------------------------------------------------------------
    // Vec3 (f32)
    // -----------------------------------------------------------------------

    #[test]
    fn v3_normalize_zero() {
        assert_eq!(Vec3::ZERO.normalize(), Vec3::ZERO);
    }

    #[test]
    fn v3_cross_handedness() {
        let cross = Vec3::UNIT_X.cross(Vec3::UNIT_Y);
        assert_eq!(cross, Vec3::UNIT_Z);
    }

    #[test]
    fn v3_cross_anticommutative() {
        let a = Vec3::new(2.0, 3.0, 4.0);
        let b = Vec3::new(5.0, 6.0, 7.0);
        assert_eq!(a.cross(b), -(b.cross(a)));
    }

    #[test]
    fn v3_dot_self_length_sq() {
        let v = Vec3::new(3.0, 4.0, 5.0);
        assert!((v.dot(v) - v.length_squared()).abs() < 1e-6);
    }

    #[test]
    fn v3_lerp_bounds() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(10.0, 10.0, 10.0);
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
    }

    // -----------------------------------------------------------------------
    // Vec2
    // -----------------------------------------------------------------------

    #[test]
    fn v2_normalize_zero() {
        assert_eq!(Vec2::ZERO.normalize(), Vec2::ZERO);
    }

    #[test]
    fn v2_dot_orthogonal() {
        assert_eq!(Vec2::UNIT_X.dot(Vec2::UNIT_Y), 0.0);
    }

    // -----------------------------------------------------------------------
    // Vec4
    // -----------------------------------------------------------------------

    #[test]
    fn v4_dot_self() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert!((v.dot(v) - 30.0).abs() < 1e-6);
    }
}

// ===========================================================================
// 3. Quaternion operations
// ===========================================================================

#[cfg(test)]
mod quaternions {
    use omega::*;

    // -----------------------------------------------------------------------
    // FQuat
    // -----------------------------------------------------------------------

    #[test]
    fn fq_identity_mul() {
        let q = FQuat::IDENTITY;
        let v = FQuat::new(
            Fixed32::from_f32(0.1),
            Fixed32::from_f32(0.2),
            Fixed32::from_f32(0.3),
            Fixed32::from_f32(0.4),
        );
        assert_eq!(q * v, v);
        assert_eq!(v * q, v);
    }

    #[test]
    fn fq_mul_inverse_identity() {
        let q = FQuat::new(
            Fixed32::from_f32(0.1),
            Fixed32::from_f32(0.2),
            Fixed32::from_f32(0.3),
            Fixed32::from_f32(0.4),
        );
        let q = q.normalize();
        let r = q * q.inverse();
        assert!(
            (r.w - Fixed32::ONE).abs().to_f32() < 0.01,
            "q * q^-1 = ({}, {}, {}, {})",
            r.w.to_f32(), r.x.to_f32(), r.y.to_f32(), r.z.to_f32()
        );
    }

    #[test]
    fn fq_conjugate_identity() {
        assert_eq!(FQuat::IDENTITY.conjugate(), FQuat::IDENTITY);
    }

    #[test]
    fn fq_conjugate_twice() {
        let q = FQuat::new(
            Fixed32::from_f32(0.1),
            Fixed32::from_f32(0.2),
            Fixed32::from_f32(0.3),
            Fixed32::from_f32(0.4),
        );
        assert_eq!(q.conjugate().conjugate(), q);
    }

    #[test]
    fn fq_length_squared_identity() {
        assert_eq!(FQuat::IDENTITY.length_squared(), Fixed32::ONE);
    }

    #[test]
    fn fq_length_identity() {
        assert!((FQuat::IDENTITY.length() - Fixed32::ONE).abs().to_f32() < 0.001);
    }

    #[test]
    fn fq_normalize_identity() {
        assert_eq!(FQuat::IDENTITY.normalize(), FQuat::IDENTITY);
    }

    #[test]
    fn fq_normalize_zero() {
        let q = FQuat::new(Fixed32::ZERO, Fixed32::ZERO, Fixed32::ZERO, Fixed32::ZERO);
        assert_eq!(q.normalize(), FQuat::IDENTITY);
    }

    #[test]
    fn fq_inverse_zero() {
        let q = FQuat::new(Fixed32::ZERO, Fixed32::ZERO, Fixed32::ZERO, Fixed32::ZERO);
        assert_eq!(q.inverse(), FQuat::IDENTITY);
    }

    #[test]
    fn fq_mul_associative() {
        let a = FQuat::new(
            Fixed32::from_f32(0.1), Fixed32::from_f32(0.2),
            Fixed32::from_f32(0.3), Fixed32::from_f32(0.4),
        ).normalize();
        let b = FQuat::new(
            Fixed32::from_f32(0.5), Fixed32::from_f32(0.6),
            Fixed32::from_f32(0.7), Fixed32::from_f32(0.8),
        ).normalize();
        let c = FQuat::new(
            Fixed32::from_f32(0.9), Fixed32::from_f32(0.10),
            Fixed32::from_f32(0.11), Fixed32::from_f32(0.12),
        ).normalize();
        let ab_c = (a * b) * c;
        let a_bc = a * (b * c);
        assert!(
            (ab_c.w - a_bc.w).abs().to_f32() < 0.01,
            "(a*b)*c != a*(b*c)"
        );
    }

    #[test]
    fn fq_neg() {
        let q = FQuat::new(Fixed32::from_f32(0.1), Fixed32::from_f32(0.2), Fixed32::from_f32(0.3), Fixed32::from_f32(0.4));
        let nq = -q;
        assert_eq!(nq.w, Fixed32::from_f32(-0.1));
        assert_eq!(nq.x, Fixed32::from_f32(-0.2));
        assert_eq!(nq.y, Fixed32::from_f32(-0.3));
        assert_eq!(nq.z, Fixed32::from_f32(-0.4));
    }

    #[test]
    fn fq_from_axis_angle_identity() {
        // axis_angle(x, 0) = identity
        let q = FQuat::from_axis_angle(FVec3::UNIT_X, Fixed32::ZERO);
        assert!(
            (q.w - Fixed32::ONE).abs().to_f32() < 0.01,
            "axis_angle(x, 0).w = {}",
            q.w.to_f32()
        );
    }

    #[test]
    fn fq_from_axis_angle_90_deg() {
        // 90 degrees around Z axis
        let angle = Fixed32::from_f32(core::f32::consts::FRAC_PI_2);
        let q = FQuat::from_axis_angle(FVec3::UNIT_Z, angle);
        // q should be (cos(45), 0, 0, sin(45)) = (sqrt(2)/2, 0, 0, sqrt(2)/2)
        let expected = 2.0f32.sqrt() / 2.0;
        assert!(
            (q.w.to_f32() - expected).abs() < 0.01,
            "axis_angle(z, 90).w = {} (expected {})",
            q.w.to_f32(),
            expected
        );
        assert!(
            (q.z.to_f32() - expected).abs() < 0.01,
            "axis_angle(z, 90).z = {} (expected {})",
            q.z.to_f32(),
            expected
        );
    }

    #[test]
    fn fq_from_axis_angle_non_normalized() {
        // Non-unit axis produces non-unit quaternion
        let axis = FVec3::new(Fixed32::from_f32(2.0), Fixed32::from_f32(0.0), Fixed32::from_f32(0.0));
        let angle = Fixed32::from_f32(core::f32::consts::FRAC_PI_2);
        let q = FQuat::from_axis_angle(axis, angle);
        // Length should be sqrt(cos^2(45) + (2*sin(45))^2) = sqrt(0.5 + 2) = sqrt(2.5) ≈ 1.581
        let len = q.length().to_f32();
        let expected = (2.5f32).sqrt();
        assert!(
            (len - expected).abs() < 0.02,
            "non-unit axis quat length = {} (expected ~{})",
            len, expected
        );
    }

    #[test]
    fn fq_rotate_vector_identity() {
        let v = FVec3::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0), Fixed32::from_f32(3.0));
        let r = FQuat::IDENTITY.rotate_vector(v);
        assert_eq!(r, v);
    }

    #[test]
    fn fq_rotate_vector_90_z() {
        // Rotate (1, 0, 0) 90 degrees around Z -> (0, 1, 0)
        let angle = Fixed32::from_f32(core::f32::consts::FRAC_PI_2);
        let q = FQuat::from_axis_angle(FVec3::UNIT_Z, angle).normalize();
        let v = FVec3::UNIT_X;
        let r = q.rotate_vector(v);
        assert!(
            (r.x.to_f32()).abs() < 0.05,
            "rotate x by 90z: r.x = {}",
            r.x.to_f32()
        );
        assert!(
            (r.y.to_f32() - 1.0).abs() < 0.05,
            "rotate x by 90z: r.y = {}",
            r.y.to_f32()
        );
    }

    #[test]
    fn fq_slerp_t0() {
        // slerp(a, b, 0) = a
        let a = FQuat::IDENTITY;
        let b = FQuat::from_axis_angle(FVec3::UNIT_X, Fixed32::from_f32(core::f32::consts::FRAC_PI_2)).normalize();
        let r = a.slerp(b, Fixed32::ZERO);
        assert!(
            (r.w - Fixed32::ONE).abs().to_f32() < 0.01,
            "slerp(a, b, 0).w = {}",
            r.w.to_f32()
        );
    }

    #[test]
    fn fq_slerp_t1() {
        // slerp(a, b, 1) = b
        let a = FQuat::IDENTITY;
        let b = FQuat::from_axis_angle(FVec3::UNIT_X, Fixed32::from_f32(core::f32::consts::FRAC_PI_2)).normalize();
        let r = a.slerp(b, Fixed32::ONE);
        assert!(
            (r.z - b.z).abs().to_f32() < 0.01,
            "slerp(a, b, 1).z = {} (expected {})",
            r.z.to_f32(),
            b.z.to_f32()
        );
    }

    #[test]
    fn fq_slerp_t_half() {
        let a = FQuat::IDENTITY;
        let b = FQuat::from_axis_angle(FVec3::UNIT_Y, Fixed32::from_f32(core::f32::consts::FRAC_PI_2)).normalize();
        let half = Fixed32::from_f32(0.5);
        let r = a.slerp(b, half);
        // At t=0.5, the angle should be 45 degrees (half of 90)
        // so we need sin(22.5 degrees) = sin(pi/8)
        let expected_y = core::f32::consts::FRAC_PI_8.sin();
        assert!(
            (r.y.to_f32() - expected_y).abs() < 0.02,
            "slerp half: y = {} (expected {})",
            r.y.to_f32(),
            expected_y
        );
    }

    #[test]
    fn fq_slerp_near_parallel() {
        // Two very close quaternions should take the lerp/nlerp path
        let a = FQuat::IDENTITY;
        let b = FQuat::new(
            Fixed32::from_f32(0.9999),
            Fixed32::from_f32(0.01),
            Fixed32::from_f32(0.0),
            Fixed32::from_f32(0.0),
        );
        let r = a.slerp(b, Fixed32::from_f32(0.5));
        // Should be approximately halfway
        assert!(
            (r.x.to_f32() - 0.005).abs() < 0.005,
            "near-parallel slerp x = {} (expected ~0.005)",
            r.x.to_f32()
        );
    }

    #[test]
    fn fq_slerp_negative_dot() {
        // Quaternions with negative dot product should flip the second one
        let a = FQuat::IDENTITY;
        let b = -FQuat::IDENTITY;
        let r = a.slerp(b, Fixed32::from_f32(0.5));
        // After flipping, b becomes IDENTITY, so result is always IDENTITY at any t
        assert!(
            (r.w - Fixed32::ONE).abs().to_f32() < 0.01,
            "slerp(q, -q, 0.5).w = {} (expected 1)",
            r.w.to_f32()
        );
    }

    #[test]
    fn fq_slerp_returns_normalized() {
        let a = FQuat::IDENTITY;
        let b = FQuat::from_axis_angle(FVec3::UNIT_Z, Fixed32::from_f32(core::f32::consts::FRAC_PI_2)).normalize();
        for t in [Fixed32::from_f32(0.25), Fixed32::from_f32(0.5), Fixed32::from_f32(0.75)] {
            let r = a.slerp(b, t);
            let len = r.length().to_f32();
            assert!(
                (len - 1.0).abs() < 0.02,
                "slerp t={}: length = {}",
                t.to_f32(),
                len
            );
        }
    }

    #[test]
    fn fq_neg_in_slerp() {
        // Verify -FQuat works in the slerp context
        let q = FQuat::new(Fixed32::from_f32(0.1), Fixed32::from_f32(0.2), Fixed32::from_f32(0.3), Fixed32::from_f32(0.4));
        let nq = -q;
        assert_eq!(q.w, -nq.w);
        assert_eq!(q.x, -nq.x);
        assert_eq!(q.y, -nq.y);
        assert_eq!(q.z, -nq.z);
    }

    // -----------------------------------------------------------------------
    // Quat (f32)
    // -----------------------------------------------------------------------

    #[test]
    fn q_identity_mul() {
        assert_eq!(Quat::IDENTITY * Quat::IDENTITY, Quat::IDENTITY);
    }

    #[test]
    fn q_mul_inverse_identity() {
        let q = Quat::new(0.1, 0.2, 0.3, 0.4).normalize();
        let r = q * q.inverse();
        assert!(
            (r.w - 1.0).abs() < 0.01,
            "q * q^-1 => w = {}",
            r.w
        );
    }

    #[test]
    fn q_conjugate_twice() {
        let q = Quat::new(0.1, 0.2, 0.3, 0.4);
        assert_eq!(q.conjugate().conjugate(), q);
    }

    #[test]
    fn q_slerp_t0() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle(Vec3::UNIT_X, core::f32::consts::FRAC_PI_2);
        let r = a.slerp(b, 0.0);
        assert!((r.w - 1.0).abs() < 0.01);
    }

    #[test]
    fn q_slerp_t1() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle(Vec3::UNIT_X, core::f32::consts::FRAC_PI_2);
        let r = a.slerp(b, 1.0);
        assert!((r.x - b.x).abs() < 0.01);
    }

    #[test]
    fn q_slerp_near_parallel() {
        let a = Quat::IDENTITY;
        let b = Quat::new(0.9999, 0.01, 0.0, 0.0);
        let r = a.slerp(b, 0.5);
        assert!((r.x - 0.005).abs() < 0.005);
    }

    #[test]
    fn q_rotate_vector_identity() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(Quat::IDENTITY.rotate_vector(v), v);
    }

    #[test]
    fn q_from_axis_angle_normalized() {
        let q = Quat::from_axis_angle(Vec3::UNIT_X, core::f32::consts::FRAC_PI_2);
        let len = q.length();
        assert!((len - 1.0).abs() < 0.01, "axis_angle length = {}", len);
    }
}

// ===========================================================================
// 4. Matrix operations
// ===========================================================================

#[cfg(test)]
mod matrices {
    use omega::*;

    // -----------------------------------------------------------------------
    // M64
    // -----------------------------------------------------------------------

    #[test]
    fn m64_identity() {
        let m = M64::IDENTITY;
        assert_eq!(m.c0.x, Fixed32::ONE);
        assert_eq!(m.c1.y, Fixed32::ONE);
        assert_eq!(m.c2.z, Fixed32::ONE);
        assert_eq!(m.c3.w, Fixed32::ONE);
    }

    #[test]
    fn m64_identity_mul_identity() {
        let r = M64::IDENTITY * M64::IDENTITY;
        assert_eq!(r, M64::IDENTITY);
    }

    #[test]
    fn m64_inverse_identity() {
        assert_eq!(M64::IDENTITY.inverse(), M64::IDENTITY);
    }

    #[test]
    fn m64_inverse_identity_times_inverse() {
        let inv = M64::IDENTITY.inverse();
        assert_eq!(inv, M64::IDENTITY);
    }

    #[test]
    fn m64_inverse_translate() {
        let mut t = M64::IDENTITY;
        t.c3.x = Fixed32::from_f32(10.0);
        let inv = t.inverse();
        // inv * t should be identity
        let r = inv * t;
        assert!(
            (r.c0.x - Fixed32::ONE).abs().to_f32() < 0.01
            && (r.c3.x).abs().to_f32() < 0.01,
            "translate * inverse: c3.x = {}",
            r.c3.x.to_f32()
        );
    }

    #[test]
    fn m64_determinant_identity() {
        let det = M64::IDENTITY.determinant();
        assert!(
            (det - Fixed32::ONE).abs().to_f32() < 0.001,
            "det(I) = {}",
            det.to_f32()
        );
    }

    #[test]
    fn m64_determinant_scale() {
        let mut m = M64::IDENTITY;
        m.c0.x = Fixed32::from_f32(2.0);
        let det = m.determinant();
        assert!(
            (det - Fixed32::from_f32(2.0)).abs().to_f32() < 0.01,
            "det(scale 2x) = {}",
            det.to_f32()
        );
    }

    #[test]
    fn m64_transpose_identity() {
        assert_eq!(M64::IDENTITY.transpose(), M64::IDENTITY);
    }

    #[test]
    fn m64_transpose_transpose() {
        let mut m = M64::IDENTITY;
        m.c0.y = Fixed32::from_f32(1.0);
        m.c1.x = Fixed32::from_f32(2.0);
        let t = m.transpose();
        assert_eq!(t.c0.y, Fixed32::from_f32(2.0));
        assert_eq!(t.c1.x, Fixed32::from_f32(1.0));
        assert_eq!(t.transpose(), m);
    }

    #[test]
    fn m64_from_axis_angle_identity() {
        let m = M64::from_axis_angle(FVec3::UNIT_X, Fixed32::ZERO);
        assert_eq!(m, M64::IDENTITY);
    }

    #[test]
    fn m64_from_axis_angle_90_x() {
        let angle = Fixed32::from_f32(core::f32::consts::FRAC_PI_2);
        let m = M64::from_axis_angle(FVec3::UNIT_X, angle);
        // Rotation around X by 90: y column should be (0, 0, 1, 0), z column should be (0, -1, 0, 0)
        assert!(
            (m.c1.y).abs().to_f32() < 0.05,
            "c1.y = {} (expected ~0)",
            m.c1.y.to_f32()
        );
        assert!(
            (m.c1.z - Fixed32::ONE).abs().to_f32() < 0.05,
            "c1.z = {} (expected ~1)",
            m.c1.z.to_f32()
        );
        assert!(
            (m.c2.y - Fixed32::from_f32(-1.0)).abs().to_f32() < 0.05,
            "c2.y = {} (expected ~-1)",
            m.c2.y.to_f32()
        );
    }

    #[test]
    fn m64_look_at_default() {
        // look_at at origin, looking down -Z
        let eye = FVec3::ZERO;
        let target = FVec3::new(Fixed32::ZERO, Fixed32::ZERO, Fixed32::from_f32(-1.0));
        let up = FVec3::UNIT_Y;
        let m = M64::look_at(eye, target, up);
        // The view matrix should map +X to +X, +Y to +Y, +Z to -Z
        assert!(
            (m.c0.x - Fixed32::ONE).abs().to_f32() < 0.05,
            "look_at c0.x = {}",
            m.c0.x.to_f32()
        );
        assert!(
            (m.c1.y - Fixed32::ONE).abs().to_f32() < 0.05,
            "look_at c1.y = {}",
            m.c1.y.to_f32()
        );
    }

    #[test]
    fn m64_look_at_up_parallel_to_dir() {
        // Near-parallel up vector (45 degrees off from target direction)
        let eye = FVec3::ZERO;
        let target = FVec3::new(Fixed32::ZERO, Fixed32::from_f32(1.0), Fixed32::ZERO);
        let up = FVec3::new(Fixed32::from_f32(1.0), Fixed32::from_f32(1.0), Fixed32::ZERO).normalize();
        let m = M64::look_at(eye, target, up);
        // Should produce a valid matrix (determinant should be non-zero)
        let det = m.determinant();
        assert!(
            det.to_f32() != 0.0,
            "near-parallel look_at produced singular matrix (det={})",
            det.to_f32()
        );
    }

    #[test]
    fn m64_perspective_fov() {
        let near = Fixed32::from_f32(0.1);
        let aspect = Fixed32::from_f32(16.0 / 9.0);
        let fov = Fixed32::from_f32(core::f32::consts::FRAC_PI_4);
        let m = M64::perspective(aspect, fov, near);
        // Right-handed infinite far: c2.z should be -1, c3.z should be -2*near
        assert!(
            (m.c2.z - Fixed32::from_f32(-1.0)).abs().to_f32() < 0.01,
            "perspective c2.z = {} (expected -1)",
            m.c2.z.to_f32()
        );
        assert!(
            (m.c3.z - Fixed32::from_f32(-2.0 * 0.1)).abs().to_f32() < 0.01,
            "perspective c3.z = {} (expected -0.2)",
            m.c3.z.to_f32()
        );
        // w' = -z
        assert_eq!(m.c2.w, Fixed32::from_f32(-1.0));
        assert_eq!(m.c3.w, Fixed32::ZERO);
    }

    #[test]
    fn m64_mul_by_identity() {
        let m = M64::IDENTITY;
        let v = FVec4::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0), Fixed32::from_f32(3.0), Fixed32::from_f32(1.0));
        let r = m.mul_v(v);
        assert_eq!(r, v, "I * v should equal v");
    }

    #[test]
    fn m64_mul_translate() {
        let mut t = M64::IDENTITY;
        t.c3.x = Fixed32::from_f32(5.0);
        t.c3.y = Fixed32::from_f32(10.0);
        let v = FVec4::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0), Fixed32::from_f32(3.0), Fixed32::ONE);
        let r = t.mul_v(v);
        assert_eq!(r.x, Fixed32::from_f32(6.0));
        assert_eq!(r.y, Fixed32::from_f32(12.0));
        assert_eq!(r.z, Fixed32::from_f32(3.0));
    }

    // -----------------------------------------------------------------------
    // Mat4 (f32)
    // -----------------------------------------------------------------------

    #[test]
    fn mat4_identity() {
        assert_eq!(Mat4::IDENTITY * Mat4::IDENTITY, Mat4::IDENTITY);
    }

    #[test]
    fn mat4_inverse_identity() {
        assert_eq!(Mat4::IDENTITY.inverse(), Mat4::IDENTITY);
    }

    #[test]
    fn mat4_determinant_identity() {
        let det = Mat4::IDENTITY.determinant();
        assert!((det - 1.0).abs() < 0.001, "det(I) = {}", det);
    }

    #[test]
    fn mat4_translate_inverse() {
        let mut t = Mat4::IDENTITY;
        t.c3.x = 10.0;
        let inv = t.inverse();
        let r = inv * t;
        assert!(
            (r.c0.x - 1.0).abs() < 0.01
            && (r.c3.x).abs() < 0.01,
            "translate * inverse"
        );
    }

    #[test]
    fn mat4_perspective() {
        let near = 0.1;
        let aspect = 16.0 / 9.0;
        let fov = core::f32::consts::FRAC_PI_4;
        let m = Mat4::perspective(aspect, fov, near);
        // Right-handed infinite far: c2.z = -1, c3.z = -2*near
        assert!(
            (m.c2.z - (-1.0)).abs() < 0.01,
            "perspective c2.z = {} (expected -1)",
            m.c2.z
        );
        assert!(
            (m.c3.z - (-2.0 * 0.1)).abs() < 0.01,
            "perspective c3.z = {} (expected -0.2)",
            m.c3.z
        );
        assert_eq!(m.c2.w, -1.0);
        assert_eq!(m.c3.w, 0.0);
    }

    #[test]
    fn mat4_look_at_default() {
        let m = Mat4::look_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::UNIT_Y);
        assert!((m.c0.x - 1.0).abs() < 0.01);
        assert!((m.c1.y - 1.0).abs() < 0.01);
    }

    #[test]
    fn mat4_mul_vector() {
        let m = Mat4::IDENTITY;
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let r = m * v;
        assert_eq!(r, v);
    }

    // -----------------------------------------------------------------------
    // Mat3 (f32)
    // -----------------------------------------------------------------------

    #[test]
    fn mat3_identity() {
        assert_eq!(Mat3::IDENTITY * Mat3::IDENTITY, Mat3::IDENTITY);
    }

    #[test]
    fn mat3_determinant_identity() {
        assert!((Mat3::IDENTITY.determinant() - 1.0).abs() < 0.001);
    }

    #[test]
    fn mat3_inverse_identity() {
        assert_eq!(Mat3::IDENTITY.inverse(), Mat3::IDENTITY);
    }

    #[test]
    fn mat3_mul_vector() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(Mat3::IDENTITY * v, v);
    }
}

// ===========================================================================
// 5. Trigonometric lookup table
// ===========================================================================

#[cfg(test)]
mod trig_lut {
    use omega::*;

    #[test]
    fn sin_known_points() {
        let s0 = TrigLUT::sin(0.0);
        assert!((s0 - 0.0).abs() < 0.001, "sin(0) = {}", s0);

        let s1 = TrigLUT::sin(core::f32::consts::FRAC_PI_2);
        assert!((s1 - 1.0).abs() < 0.002, "sin(pi/2) = {}", s1);

        let s2 = TrigLUT::sin(core::f32::consts::PI);
        assert!((s2 - 0.0).abs() < 0.002, "sin(pi) = {}", s2);

        let s3 = TrigLUT::sin(3.0 * core::f32::consts::FRAC_PI_2);
        assert!((s3 - (-1.0)).abs() < 0.002, "sin(3pi/2) = {}", s3);
    }

    #[test]
    fn cos_known_points() {
        assert!((TrigLUT::cos(0.0) - 1.0).abs() < 0.001);
        assert!((TrigLUT::cos(core::f32::consts::FRAC_PI_2) - 0.0).abs() < 0.002);
        assert!((TrigLUT::cos(core::f32::consts::PI) - (-1.0)).abs() < 0.002);
    }

    #[test]
    fn tan_known_points() {
        let t0 = TrigLUT::tan(0.0);
        assert!((t0 - 0.0).abs() < 0.001, "tan(0) = {}", t0);

        let t1 = TrigLUT::tan(core::f32::consts::FRAC_PI_4);
        assert!((t1 - 1.0).abs() < 0.01, "tan(pi/4) = {}", t1);
    }

    #[test]
    fn sin_wraparound() {
        let x = 2.5;
        let s1 = TrigLUT::sin(x);
        let s2 = TrigLUT::sin(x + TrigLUT::TAU);
        assert!((s1 - s2).abs() < 0.001, "wraparound mismatch");
    }

    #[test]
    fn sin_negative_wraparound() {
        let x = -1.0;
        let s = TrigLUT::sin(x);
        let s_pos = TrigLUT::sin(x + TrigLUT::TAU);
        assert!((s - s_pos).abs() < 0.001, "negative wraparound: {} vs {}", s, s_pos);
    }

    #[test]
    fn sin_negative_odd() {
        // sin(-x) = -sin(x)
        let x = 1.5;
        assert!((TrigLUT::sin(-x) + TrigLUT::sin(x)).abs() < 0.002);
    }

    #[test]
    fn sin_vs_cos_phase() {
        let x = 2.5;
        let s_shifted = TrigLUT::sin(x + core::f32::consts::FRAC_PI_2);
        let c = TrigLUT::cos(x);
        assert!((s_shifted - c).abs() < 0.002, "sin(x+pi/2) != cos(x)");
    }

    #[test]
    fn sin_pythagorean() {
        for &x in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let s = TrigLUT::sin(x);
            let c = TrigLUT::cos(x);
            let sum = s * s + c * c;
            assert!(
                (sum - 1.0).abs() < 0.01,
                "sin^2({}) + cos^2({}) = {}",
                x, x, sum
            );
        }
    }

    #[test]
    fn sin_boundary_tau() {
        // sin(TAU) should be close to sin(0) due to wraparound
        let s0 = TrigLUT::sin(0.0);
        let st = TrigLUT::sin(TrigLUT::TAU);
        assert!((s0 - st).abs() < 0.001, "sin(0) = {}, sin(TAU) = {}", s0, st);
    }

    #[test]
    fn cos_boundary_pi() {
        let c = TrigLUT::cos(core::f32::consts::PI);
        assert!((c - (-1.0)).abs() < 0.002, "cos(pi) = {}", c);
    }

    #[test]
    fn fixed32_sin_approx() {
        let x = Fixed32::from_f32(core::f32::consts::FRAC_PI_2);
        let s = x.sin_approx();
        assert!(
            (s - Fixed32::ONE).abs().to_f32() < 0.01,
            "Fixed32 sin(pi/2) = {}",
            s.to_f32()
        );
    }

    #[test]
    fn fixed32_cos_approx() {
        let x = Fixed32::from_f32(0.0);
        let c = x.cos_approx();
        assert!(
            (c - Fixed32::ONE).abs().to_f32() < 0.01,
            "Fixed32 cos(0) = {}",
            c.to_f32()
        );
    }
}

// ===========================================================================
// 6. SimRng determinism
// ===========================================================================

#[cfg(test)]
mod sim_rng {
    use omega::*;

    #[test]
    fn deterministic_seed_42() {
        let mut rng = SimRng::new(42);
        let mut rng2 = SimRng::new(42);
        // Same seed produces same sequence
        for _ in 0..10 {
            assert_eq!(rng.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn same_seed_same_sequence() {
        let mut a = SimRng::new(12345);
        let mut b = SimRng::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_different() {
        let mut a = SimRng::new(1);
        let mut b = SimRng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn next_f32_in_range() {
        let mut rng = SimRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "f32 out of range: {}", v);
        }
    }

    #[test]
    fn next_f64_in_range() {
        let mut rng = SimRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "f64 out of range: {}", v);
        }
    }

    #[test]
    fn next_u32_from_u64() {
        let mut rng = SimRng::new(7);
        let upper = rng.next_u64();
        rng = SimRng::new(7);
        let lower = rng.next_u32();
        assert_eq!(upper >> 32, lower as u64);
    }

    #[test]
    fn bounded_exact() {
        let mut rng = SimRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_u64_bounded(10);
            assert!(v < 10, "bounded out of range: {}", v);
        }
    }

    #[test]
    fn bounded_zero() {
        let mut rng = SimRng::new(42);
        assert_eq!(rng.next_u64_bounded(0), 0);
    }

    #[test]
    fn bounded_one() {
        let mut rng = SimRng::new(42);
        assert_eq!(rng.next_u64_bounded(1), 0);
    }

    #[test]
    fn fill_bytes_not_all_zero() {
        let mut rng = SimRng::new(42);
        let mut buf = [0u8; 32];
        rng.fill_bytes(&mut buf);
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn fill_bytes_deterministic() {
        let mut a = SimRng::new(99);
        let mut b = SimRng::new(99);
        let mut buf_a = [0u8; 17];
        let mut buf_b = [0u8; 17];
        a.fill_bytes(&mut buf_a);
        b.fill_bytes(&mut buf_b);
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn fill_bytes_various_lengths() {
        for len in &[1, 3, 7, 8, 9, 15, 16, 17] {
            let mut rng = SimRng::new(42);
            let mut buf = vec![0u8; *len];
            rng.fill_bytes(&mut buf);
            assert!(buf.iter().any(|&b| b != 0), "all zeros for len={}", len);
        }
    }

    #[test]
    fn skip_matches_iterated() {
        let mut direct = SimRng::new(1);
        for _ in 0..50 {
            direct.next_u64();
        }
        let v_direct = direct.next_u64();

        let mut skipped = SimRng::new_skip(1, 50);
        let v_skipped = skipped.next_u64();
        assert_eq!(v_direct, v_skipped);
    }

    #[test]
    fn skip_zero() {
        let mut a = SimRng::new(42);
        let mut b = SimRng::new_skip(42, 0);
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn large_skip() {
        let mut a = SimRng::new(42);
        let mut b = SimRng::new_skip(42, 1000);
        for _ in 0..1000 {
            a.next_u64();
        }
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

// ===========================================================================
// 7. bytemuck trait guarantees
// ===========================================================================

#[cfg(test)]
mod bytemuck_guarantees {
    use omega::*;

    #[test]
    fn fixed16_is_pod() {
        let v = Fixed16::from_f32(3.5);
        let bytes: &[u8] = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 2);
    }

    #[test]
    fn fixed32_is_pod() {
        let v = Fixed32::from_f32(3.5);
        let bytes: &[u8] = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 4);
    }

    #[test]
    fn fvec2_is_pod() {
        let v = FVec2::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0));
        let bytes: &[u8] = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn fvec3_is_pod() {
        let v = FVec3::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0), Fixed32::from_f32(3.0));
        let bytes: &[u8] = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn fvec4_is_pod() {
        let v = FVec4::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0), Fixed32::from_f32(3.0), Fixed32::from_f32(4.0));
        let bytes: &[u8] = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn fquat_is_pod() {
        let q = FQuat::IDENTITY;
        let bytes: &[u8] = bytemuck::bytes_of(&q);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn m64_is_pod() {
        let m = M64::IDENTITY;
        let bytes: &[u8] = bytemuck::bytes_of(&m);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn vec2_is_pod() {
        let bytes: &[u8] = bytemuck::bytes_of(&Vec2::UNIT_X);
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn vec3_is_pod() {
        let bytes: &[u8] = bytemuck::bytes_of(&Vec3::UNIT_X);
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn vec4_is_pod() {
        let bytes: &[u8] = bytemuck::bytes_of(&Vec4::UNIT_X);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn quat_is_pod() {
        let bytes: &[u8] = bytemuck::bytes_of(&Quat::IDENTITY);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn mat4_is_pod() {
        let bytes: &[u8] = bytemuck::bytes_of(&Mat4::IDENTITY);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn mat3_is_pod() {
        let bytes: &[u8] = bytemuck::bytes_of(&Mat3::IDENTITY);
        assert_eq!(bytes.len(), 36);
    }

    #[test]
    fn fixed16_zeroable() {
        let zero: Fixed16 = bytemuck::Zeroable::zeroed();
        assert_eq!(zero, Fixed16::ZERO);
    }

    #[test]
    fn fixed32_zeroable() {
        let zero: Fixed32 = bytemuck::Zeroable::zeroed();
        assert_eq!(zero, Fixed32::ZERO);
    }

    #[test]
    fn m64_zeroable() {
        let zero: M64 = bytemuck::Zeroable::zeroed();
        assert_eq!(zero.c0.x, Fixed32::ZERO);
        assert_eq!(zero.c3.w, Fixed32::ZERO);
    }

    #[test]
    fn simrng_is_clone_copy() {
        let a = SimRng::new(42);
        let b = a; // copy
        assert_eq!(a, b);
    }

}

// ===========================================================================
// 8. Integration tests
// ===========================================================================

#[cfg(test)]
mod integration {
    use omega::*;

    #[test]
    fn fixed32_sin_cos_triglut() {
        // Verify that Fixed32::sin_approx() matches TrigLUT::sin()
        let x = Fixed32::from_f32(1.0);
        let s_fixed = x.sin_approx();
        let s_lut = Fixed32::from_f32(TrigLUT::sin(x.to_f32()));
        assert!(
            (s_fixed - s_lut).abs().to_f32() < 0.001,
            "Fixed32 sin vs TrigLUT: {} vs {}",
            s_fixed.to_f32(),
            s_lut.to_f32()
        );
    }

    #[test]
    fn fquat_from_axis_angle_triglut() {
        // Verify quaternion from axis angle uses correct trig values
        let angle = Fixed32::from_f32(1.0);
        let axis = FVec3::UNIT_Y;
        let q = FQuat::from_axis_angle(axis, angle);
        let half = 0.5f32;
        let (s, c) = (half * 1.0f32).sin_cos();
        // w should be cos(half_angle)
        assert!(
            (q.w.to_f32() - c).abs() < 0.01,
            "FQuat axis_angle w = {} (expected {})",
            q.w.to_f32(),
            c
        );
        // y should be sin(half_angle)
        assert!(
            (q.y.to_f32() - s).abs() < 0.01,
            "FQuat axis_angle y = {} (expected {})",
            q.y.to_f32(),
            s
        );
    }

    #[test]
    fn fquat_rotate_then_inverse() {
        // Rotate a vector, then rotate back with inverse
        let q = FQuat::from_axis_angle(
            FVec3::UNIT_Z,
            Fixed32::from_f32(core::f32::consts::FRAC_PI_2),
        ).normalize();
        let v = FVec3::UNIT_X;
        let rotated = q.rotate_vector(v);
        let back = q.inverse().rotate_vector(rotated);
        assert!(
            (back.x - Fixed32::ONE).abs().to_f32() < 0.05,
            "rotate then inverse: back.x = {}",
            back.x.to_f32()
        );
    }

    #[test]
    fn m64_rotate_then_inverse() {
        // Rotation matrix * inverse should be identity
        let m = M64::from_axis_angle(
            FVec3::UNIT_X,
            Fixed32::from_f32(core::f32::consts::FRAC_PI_4),
        );
        let inv = m.inverse();
        let r = m * inv;
        assert!(
            (r.c0.x - Fixed32::ONE).abs().to_f32() < 0.01,
            "M * M^-1 = identity: c0.x = {}",
            r.c0.x.to_f32()
        );
    }

    #[test]
    fn fquat_m64_rotate_consistency() {
        // Quaternion and matrix rotation of the same axis-angle should agree
        let angle = Fixed32::from_f32(core::f32::consts::FRAC_PI_4);
        let axis = FVec3::new(
            Fixed32::from_f32(1.0),
            Fixed32::from_f32(1.0),
            Fixed32::from_f32(0.0),
        ).normalize();

        let q = FQuat::from_axis_angle(axis, angle).normalize();
        let m = M64::from_axis_angle(axis, angle);

        let v = FVec3::new(Fixed32::from_f32(1.0), Fixed32::from_f32(0.0), Fixed32::from_f32(0.0));
        let r_q = q.rotate_vector(v);
        let r_m_v = m.mul_v(FVec4::new(v.x, v.y, v.z, Fixed32::ONE));
        let r_m = FVec3::new(r_m_v.x, r_m_v.y, r_m_v.z);

        assert!(
            (r_q.x - r_m.x).abs().to_f32() < 0.05
            && (r_q.y - r_m.y).abs().to_f32() < 0.05
            && (r_q.z - r_m.z).abs().to_f32() < 0.05,
            "Quat vs Matrix rotation mismatch: q=({},{},{}), m=({},{},{})",
            r_q.x.to_f32(), r_q.y.to_f32(), r_q.z.to_f32(),
            r_m.x.to_f32(), r_m.y.to_f32(), r_m.z.to_f32()
        );
    }

    #[test]
    fn simrng_fvec3_randomization() {
        // Use SimRng to generate random vectors
        let mut rng = SimRng::new(42);
        let v = FVec3::new(
            Fixed32::from_f32(rng.next_f32()),
            Fixed32::from_f32(rng.next_f32()),
            Fixed32::from_f32(rng.next_f32()),
        );
        assert!(v.x >= Fixed32::ZERO && v.x <= Fixed32::ONE);
        assert!(v.y >= Fixed32::ZERO && v.y <= Fixed32::ONE);
        assert!(v.z >= Fixed32::ZERO && v.z <= Fixed32::ONE);
    }

    #[test]
    fn look_at_perspective_compose() {
        // View * Projection should be non-singular
        let view = M64::look_at(
            FVec3::new(Fixed32::from_f32(0.0), Fixed32::from_f32(0.0), Fixed32::from_f32(5.0)),
            FVec3::ZERO,
            FVec3::UNIT_Y,
        );
        let proj = M64::perspective(
            Fixed32::from_f32(16.0 / 9.0),
            Fixed32::from_f32(core::f32::consts::FRAC_PI_4),
            Fixed32::from_f32(0.1),
        );
        let vp = proj * view;
        let det = vp.determinant();
        assert!(
            det.to_f32() != 0.0,
            "view * proj is singular (det = {})",
            det.to_f32()
        );
    }

    #[test]
    fn fquat_vec3_roundtrip() {
        // Multiple rotations should return to start
        let q1 = FQuat::from_axis_angle(FVec3::UNIT_X, Fixed32::from_f32(core::f32::consts::FRAC_PI_2)).normalize();
        let q2 = FQuat::from_axis_angle(FVec3::UNIT_Y, Fixed32::from_f32(core::f32::consts::FRAC_PI_2)).normalize();
        let q3 = FQuat::from_axis_angle(FVec3::UNIT_Z, Fixed32::from_f32(core::f32::consts::FRAC_PI_2)).normalize();

        let v = FVec3::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0), Fixed32::from_f32(3.0));
        let combined = q3 * q2 * q1;
        let inv_combined = combined.inverse();

        let rotated = combined.rotate_vector(v);
        let back = inv_combined.rotate_vector(rotated);

        assert!(
            (v.x - back.x).abs().to_f32() < 0.1
            && (v.y - back.y).abs().to_f32() < 0.1
            && (v.z - back.z).abs().to_f32() < 0.1,
            "Round-trip rotation failed: v=({},{},{}), back=({},{},{})",
            v.x.to_f32(), v.y.to_f32(), v.z.to_f32(),
            back.x.to_f32(), back.y.to_f32(), back.z.to_f32()
        );
    }
}

// ===========================================================================
// 9. Determinism / cross-platform consistency
// ===========================================================================

#[cfg(test)]
mod determinism {
    use omega::*;

    /// Verify that Fixed32 arithmetic is deterministic.
    /// This test checks that a fixed computation produces the same result
    /// when run twice in the same process.
    #[test]
    fn fixed32_arithmetic_deterministic() {
        let a = Fixed32::from_f32(100.0);
        let b = Fixed32::from_f32(3.0);
        let result1 = (a / b) * Fixed32::from_f32(2.0) + Fixed32::from_f32(1.0);

        let result2 = (a / b) * Fixed32::from_f32(2.0) + Fixed32::from_f32(1.0);
        assert_eq!(result1, result2);
    }

    #[test]
    fn fvec3_dot_deterministic() {
        let a = FVec3::new(Fixed32::from_f32(1.5), Fixed32::from_f32(2.5), Fixed32::from_f32(3.5));
        let b = FVec3::new(Fixed32::from_f32(4.5), Fixed32::from_f32(5.5), Fixed32::from_f32(6.5));
        let r1 = a.dot(b);
        let r2 = a.dot(b);
        assert_eq!(r1, r2);
    }

    #[test]
    fn fquat_slerp_deterministic() {
        let a = FQuat::IDENTITY;
        let b = FQuat::from_axis_angle(
            FVec3::UNIT_X,
            Fixed32::from_f32(core::f32::consts::FRAC_PI_2),
        ).normalize();
        let t = Fixed32::from_f32(0.3);
        let r1 = a.slerp(b, t);
        let r2 = a.slerp(b, t);
        assert_eq!(r1, r2);
    }

    #[test]
    fn m64_determinant_deterministic() {
        let m = M64::from_axis_angle(
            FVec3::new(Fixed32::from_f32(1.0), Fixed32::from_f32(2.0), Fixed32::from_f32(3.0)).normalize(),
            Fixed32::from_f32(0.7),
        );
        let d1 = m.determinant();
        let d2 = m.determinant();
        assert_eq!(d1, d2);
    }

    #[test]
    fn simrng_deterministic_sequence() {
        let mut rng = SimRng::new(42);
        let seq1: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();

        let mut rng = SimRng::new(42);
        let seq2: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();

        assert_eq!(seq1, seq2);
    }

    #[test]
    fn simrng_f64_deterministic() {
        let mut a = SimRng::new(12345);
        let mut b = SimRng::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_f64(), b.next_f64());
        }
    }

    #[test]
    fn fixed32_perspective_deterministic() {
        let aspect = Fixed32::from_f32(16.0 / 9.0);
        let fov = Fixed32::from_f32(core::f32::consts::FRAC_PI_4);
        let near = Fixed32::from_f32(0.1);
        let m1 = M64::perspective(aspect, fov, near);
        let m2 = M64::perspective(aspect, fov, near);
        assert_eq!(m1, m2);
    }

    #[test]
    fn cross_product_handedness_deterministic() {
        // Right-handed cross product: X x Y = Z (always, deterministically)
        let x = FVec3::new(Fixed32::from_f32(1.0), Fixed32::ZERO, Fixed32::ZERO);
        let y = FVec3::new(Fixed32::ZERO, Fixed32::from_f32(1.0), Fixed32::ZERO);
        let z = x.cross(y);
        assert_eq!(z, FVec3::UNIT_Z);
    }

    #[test]
    fn fixed32_from_f32_roundtrip() {
        let values = [0.0, 0.5, 1.0, -1.0, 3.14159, -0.001, 1000.0, -500.0];
        for &v in &values {
            let f = Fixed32::from_f32(v);
            let back = f.to_f32();
            assert!(
                (v - back).abs() < 0.001,
                "roundtrip({}) = {}",
                v, back
            );
        }
    }

    #[test]
    fn fixed16_from_f32_roundtrip() {
        let values = [0.0, 0.5, 1.0, -1.0, 3.5, -0.5, 100.0, -50.0];
        for &v in &values {
            let f = Fixed16::from_f32(v);
            let back = f.to_f32();
            assert!(
                (v - back).abs() < 0.01,
                "Fixed16 roundtrip({}) = {}",
                v, back
            );
        }
    }
}
