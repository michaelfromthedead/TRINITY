//! Whitebox tests for contract attribute parsing.

use trinity_contracts::contract;

// ==================== Outer Attributes ====================

#[contract]
#[requires(x > 0)]
fn with_requires(x: i32) -> i32 {
    x * 2
}

#[test]
fn test_outer_requires() {
    assert_eq!(with_requires(5), 10);
}

#[contract]
#[ensures(*result > 0)]
fn with_ensures(x: i32) -> i32 {
    x.abs() + 1
}

#[test]
fn test_outer_ensures() {
    assert_eq!(with_ensures(-5), 6);
}

#[contract]
#[requires(a > 0)]
#[requires(b > 0)]
#[ensures(*result > a)]
#[ensures(*result > b)]
fn with_multiple(a: i32, b: i32) -> i32 {
    a + b
}

#[test]
fn test_multiple_attrs() {
    assert_eq!(with_multiple(3, 5), 8);
}

// ==================== No Attributes ====================

#[contract]
fn no_contract_attrs(x: i32) -> i32 {
    x + 1
}

#[test]
fn test_no_attrs() {
    assert_eq!(no_contract_attrs(5), 6);
}

// ==================== No Return ====================

#[contract]
fn no_return_value() {
    let _ = 1 + 1;
}

#[test]
fn test_no_return() {
    no_return_value();
}

// ==================== Complex Expressions ====================

#[contract]
#[requires(x.is_positive())]
fn with_method_call(x: i32) -> i32 {
    x
}

#[test]
fn test_method_call_in_requires() {
    assert_eq!(with_method_call(5), 5);
}

#[contract]
#[requires(x > 0 && y > 0)]
fn with_logical_and(x: i32, y: i32) -> i32 {
    x + y
}

#[test]
fn test_logical_and() {
    assert_eq!(with_logical_and(3, 4), 7);
}

#[contract]
#[requires(x >= 0 || y >= 0)]
fn with_logical_or(x: i32, y: i32) -> i32 {
    x.max(y)
}

#[test]
fn test_logical_or() {
    assert_eq!(with_logical_or(-1, 5), 5);
}

// ==================== Reference Parameters ====================

#[contract]
#[requires(!s.is_empty())]
fn with_ref_param(s: &str) -> usize {
    s.len()
}

#[test]
fn test_ref_param() {
    assert_eq!(with_ref_param("hello"), 5);
}

// ==================== Generic Parameters ====================

#[contract]
fn generic_func<T: Clone>(x: T) -> T {
    x.clone()
}

#[test]
fn test_generic() {
    assert_eq!(generic_func(42), 42);
    assert_eq!(generic_func("test"), "test");
}

// ==================== Async-like (but sync) ====================

#[contract]
#[requires(n > 0)]
fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

#[test]
fn test_recursive() {
    assert_eq!(factorial(5), 120);
}
