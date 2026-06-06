//! Blackbox tests for contract parsing with complex scenarios.

use trinity_contracts::contract;

// Test: Math operations with contracts
mod math_ops {
    use super::*;

    #[contract]
    #[requires(x >= 0.0)]
    #[ensures(*result >= 0.0)]
    pub fn sqrt_approx(x: f64) -> f64 {
        x.sqrt()
    }

    #[contract]
    #[requires(b != 0)]
    pub fn safe_divide(a: i32, b: i32) -> i32 {
        a / b
    }

    #[test]
    fn test_sqrt_approx() {
        assert!((sqrt_approx(4.0) - 2.0).abs() < 0.001);
        assert!((sqrt_approx(9.0) - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_safe_divide() {
        assert_eq!(safe_divide(10, 2), 5);
        assert_eq!(safe_divide(-10, 2), -5);
    }
}

// Test: String operations with contracts
mod string_ops {
    use super::*;

    #[contract]
    #[requires(!s.is_empty())]
    #[ensures(*result > 0)]
    pub fn string_len(s: &str) -> usize {
        s.len()
    }

    #[contract]
    #[requires(n > 0)]
    pub fn repeat_char(c: char, n: usize) -> String {
        std::iter::repeat(c).take(n).collect()
    }

    #[test]
    fn test_string_len() {
        assert_eq!(string_len("hello"), 5);
        assert_eq!(string_len("a"), 1);
    }

    #[test]
    fn test_repeat_char() {
        assert_eq!(repeat_char('x', 3), "xxx");
    }
}

// Test: Vec operations with contracts
mod vec_ops {
    use super::*;

    #[contract]
    #[requires(!v.is_empty())]
    pub fn first_element(v: &[i32]) -> i32 {
        v[0]
    }

    #[contract]
    #[requires(!v.is_empty())]
    pub fn last_element(v: &[i32]) -> i32 {
        v[v.len() - 1]
    }

    #[test]
    fn test_first_element() {
        assert_eq!(first_element(&[1, 2, 3]), 1);
    }

    #[test]
    fn test_last_element() {
        assert_eq!(last_element(&[1, 2, 3]), 3);
    }
}

// Test: Nested contracts
mod nested {
    use super::*;

    #[contract]
    #[requires(x > 0)]
    fn outer(x: i32) -> i32 {
        inner(x * 2)
    }

    #[contract]
    #[requires(x > 0)]
    fn inner(x: i32) -> i32 {
        x + 1
    }

    #[test]
    fn test_nested_calls() {
        assert_eq!(outer(5), 11); // inner(5 * 2) = inner(10) = 11
    }
}

// Test: Complex ensures
mod complex_ensures {
    use super::*;

    #[contract]
    #[requires(v.len() >= 2)]
    #[ensures(result.len() >= 2)]
    fn sort_vec(v: Vec<i32>) -> Vec<i32> {
        let mut sorted = v;
        sorted.sort();
        sorted
    }

    #[contract]
    #[ensures(result.is_some())]
    fn always_some(x: i32) -> Option<i32> {
        Some(x + 1)
    }

    #[test]
    fn test_sort_vec() {
        let result = sort_vec(vec![3, 1, 2]);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_always_some() {
        assert_eq!(always_some(5), Some(6));
    }
}
