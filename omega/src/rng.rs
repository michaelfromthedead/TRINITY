// Deterministic pseudo-random number generator using splitmix64.
//
// SimRng provides deterministic random sequences across platforms,
// suitable for simulation and procedural generation.

/// Splitmix64-based deterministic PRNG.
///
/// Cross-platform deterministic sequence from a given seed.
/// Suitable for simulation, not cryptography.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SimRng {
    state: u64,
}

impl SimRng {
    /// Create a new RNG from a seed.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Create a new RNG from a seed, skipping `n` outputs.
    #[inline]
    pub fn new_skip(seed: u64, n: u64) -> Self {
        let mut rng = Self::new(seed);
        for _ in 0..n {
            rng.next_u64();
        }
        rng
    }

    /// Generate the next u64 using splitmix64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut z = self.state.wrapping_add(0x9e3779b97f4a7c15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generate the next u32 (upper 32 bits of next_u64).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generate the next f32 in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Generate the next f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Generate a u64 in [0, bound).
    ///
    /// Uses Lemire's unbiased rejection method with 128-bit math.
    #[inline]
    pub fn next_u64_bounded(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        let mut x = self.next_u64();
        let mut m = (x as u128) * (bound as u128);
        let mut l = m as u64;
        if l < bound {
            let t = (!bound).wrapping_add(1) % bound;
            while l < t {
                x = self.next_u64();
                m = (x as u128) * (bound as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Fill a byte slice with random data.
    #[inline]
    pub fn fill_bytes(&mut self, buf: &mut [u8]) {
        let mut i = 0;
        while i < buf.len() {
            let val = self.next_u64();
            let remaining = buf.len() - i;
            let bytes_to_copy = remaining.min(8);
            let src = &val.to_le_bytes()[..bytes_to_copy];
            buf[i..i + bytes_to_copy].copy_from_slice(src);
            i += bytes_to_copy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sequence() {
        let mut rng = SimRng::new(42);
        let a = rng.next_u64();
        let b = rng.next_u64();
        let c = rng.next_u64();

        // Verify determinism: same seed = same values
        let mut rng2 = SimRng::new(42);
        assert_eq!(a, rng2.next_u64());
        assert_eq!(b, rng2.next_u64());
        assert_eq!(c, rng2.next_u64());

        // Verify non-zeroness and diversity
        assert!(a != 0, "first value should be non-zero");
        assert!(a != b, "sequential values should differ");
    }

    #[test]
    fn different_seeds_different() {
        let mut a = SimRng::new(1);
        let mut b = SimRng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn same_seed_same() {
        let mut a = SimRng::new(12345);
        let mut b = SimRng::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn next_f32_range() {
        let mut rng = SimRng::new(99);
        for _ in 0..1000 {
            let val = rng.next_f32();
            assert!(val >= 0.0 && val < 1.0, "f32 out of range: {}", val);
        }
    }

    #[test]
    fn next_f64_range() {
        let mut rng = SimRng::new(99);
        for _ in 0..1000 {
            let val = rng.next_f64();
            assert!(val >= 0.0 && val < 1.0, "f64 out of range: {}", val);
        }
    }

    #[test]
    fn next_u32_from_u64() {
        let mut rng = SimRng::new(7);
        let upper = rng.next_u64();
        rng = SimRng::new(7);
        let lower = rng.next_u32() as u64;
        assert_eq!(upper >> 32, lower);
    }

    #[test]
    fn bounded_range() {
        let mut rng = SimRng::new(42);
        for _ in 0..1000 {
            let bound = 10;
            let val = rng.next_u64_bounded(bound);
            assert!(val < bound, "bounded u64 out of range: {}", val);
        }
    }

    #[test]
    fn fill_bytes_length() {
        let mut rng = SimRng::new(42);
        let mut buf = [0u8; 17];
        rng.fill_bytes(&mut buf);
        // Verify not all zero
        assert!(buf.iter().any(|&b| b != 0), "fill_bytes produced all zeros");
        // Verify deterministic
        let mut rng2 = SimRng::new(42);
        let mut buf2 = [0u8; 17];
        rng2.fill_bytes(&mut buf2);
        assert_eq!(buf, buf2);
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
}
