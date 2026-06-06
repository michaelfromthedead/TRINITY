//! Whitebox tests for runtime contract checking.

use trinity_contracts::runtime::{
    check_ensures, check_invariant, check_requires, debug_ensures, debug_invariant,
    debug_requires, CheckKind, CheckResult, ContractChecker,
};

// ==================== CheckResult ====================

#[test]
fn test_check_result_pass() {
    let result = CheckResult::pass("x > 0", "my_func", CheckKind::Precondition);
    assert!(result.passed);
    assert!(result.message.is_none());
}

#[test]
fn test_check_result_fail() {
    let result = CheckResult::fail(
        "x > 0",
        "my_func",
        CheckKind::Precondition,
        "value was negative",
    );
    assert!(!result.passed);
    assert!(result.message.is_some());
}

// ==================== CheckKind ====================

#[test]
fn test_check_kind_display() {
    assert_eq!(format!("{}", CheckKind::Precondition), "Precondition");
    assert_eq!(format!("{}", CheckKind::Postcondition), "Postcondition");
    assert_eq!(format!("{}", CheckKind::Invariant), "Invariant");
}

// ==================== check_* functions ====================

#[test]
fn test_check_requires_pass() {
    check_requires(true, "x > 0", "test_func");
}

#[test]
#[should_panic(expected = "precondition")]
fn test_check_requires_fail() {
    check_requires(false, "x > 0", "test_func");
}

#[test]
fn test_check_ensures_pass() {
    check_ensures(true, "result > 0", "test_func");
}

#[test]
#[should_panic(expected = "postcondition")]
fn test_check_ensures_fail() {
    check_ensures(false, "result > 0", "test_func");
}

#[test]
fn test_check_invariant_pass() {
    check_invariant(true, "self.valid()", "test_func");
}

#[test]
#[should_panic(expected = "invariant")]
fn test_check_invariant_fail() {
    check_invariant(false, "self.valid()", "test_func");
}

// ==================== debug_* functions ====================

#[test]
fn test_debug_requires_pass() {
    debug_requires(true, "x > 0", "test_func");
}

#[test]
fn test_debug_ensures_pass() {
    debug_ensures(true, "result > 0", "test_func");
}

#[test]
fn test_debug_invariant_pass() {
    debug_invariant(true, "self.valid()", "test_func");
}

// ==================== ContractChecker ====================

#[test]
fn test_checker_new() {
    let checker = ContractChecker::new("my_func");
    assert!(!checker.has_violations());
}

#[test]
fn test_checker_requires_pass() {
    let mut checker = ContractChecker::new("my_func");
    checker.requires(true, "x > 0");
    assert!(!checker.has_violations());
}

#[test]
fn test_checker_requires_fail() {
    let mut checker = ContractChecker::new("my_func");
    checker.requires(false, "x > 0");
    assert!(checker.has_violations());
    assert_eq!(checker.violations().len(), 1);
}

#[test]
fn test_checker_ensures_pass() {
    let mut checker = ContractChecker::new("my_func");
    checker.ensures(true, "result > 0");
    assert!(!checker.has_violations());
}

#[test]
fn test_checker_ensures_fail() {
    let mut checker = ContractChecker::new("my_func");
    checker.ensures(false, "result > 0");
    assert!(checker.has_violations());
}

#[test]
fn test_checker_multiple_violations() {
    let mut checker = ContractChecker::new("my_func");
    checker
        .requires(false, "x > 0")
        .requires(false, "y > 0")
        .ensures(false, "result > 0");
    assert_eq!(checker.violations().len(), 3);
}

#[test]
fn test_checker_assert_valid_pass() {
    let checker = ContractChecker::new("my_func");
    checker.assert_valid();
}

#[test]
#[should_panic(expected = "Contract violations")]
fn test_checker_assert_valid_fail() {
    let mut checker = ContractChecker::new("my_func");
    checker.requires(false, "x > 0");
    checker.assert_valid();
}

#[test]
fn test_checker_chaining() {
    let mut checker = ContractChecker::new("complex_func");
    checker
        .requires(true, "a > 0")
        .requires(true, "b > 0")
        .invariant(true, "self.count > 0")
        .ensures(true, "result.is_valid()");
    assert!(!checker.has_violations());
}
