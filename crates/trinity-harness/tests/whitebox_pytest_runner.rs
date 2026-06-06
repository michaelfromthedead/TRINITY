//! Whitebox tests for pytest runner.

use trinity_harness::runners::{PytestConfig, PytestResult, TestOutcome, TestResult};

// ==================== PytestConfig ====================

#[test]
fn test_config_default() {
    let config = PytestConfig::default();
    assert_eq!(config.working_dir, ".");
    assert!(config.test_path.is_none());
    assert!(config.test_name.is_none());
    assert_eq!(config.timeout_secs, 1800);
}

#[test]
fn test_config_new() {
    let config = PytestConfig::new("/path/to/tests");
    assert_eq!(config.working_dir, "/path/to/tests");
}

#[test]
fn test_config_builder() {
    let config = PytestConfig::new(".")
        .path("tests/unit")
        .test("test_foo")
        .timeout(600);

    assert_eq!(config.test_path, Some("tests/unit".to_string()));
    assert_eq!(config.test_name, Some("test_foo".to_string()));
    assert_eq!(config.timeout_secs, 600);
}

// ==================== PytestResult ====================

#[test]
fn test_result_new() {
    let result = PytestResult::new();
    assert_eq!(result.total, 0);
    assert_eq!(result.passed, 0);
    assert_eq!(result.failed, 0);
    assert_eq!(result.skipped, 0);
    assert!(result.tests.is_empty());
}

#[test]
fn test_result_add_passed() {
    let mut result = PytestResult::new();
    result.add_result(TestResult::passed("test1", 100));

    assert_eq!(result.total, 1);
    assert_eq!(result.passed, 1);
    assert_eq!(result.failed, 0);
    assert_eq!(result.total_duration_ms, 100);
}

#[test]
fn test_result_add_failed() {
    let mut result = PytestResult::new();
    result.add_result(TestResult::failed("test1", 100, "assertion error"));

    assert_eq!(result.total, 1);
    assert_eq!(result.passed, 0);
    assert_eq!(result.failed, 1);
}

#[test]
fn test_result_add_skipped() {
    let mut result = PytestResult::new();
    result.add_result(TestResult::ignored("test1"));

    assert_eq!(result.total, 1);
    assert_eq!(result.skipped, 1);
}

#[test]
fn test_result_finalize_success() {
    let mut result = PytestResult::new();
    result.add_result(TestResult::passed("test1", 100));
    result.finalize();

    assert!(result.success);
}

#[test]
fn test_result_finalize_failure() {
    let mut result = PytestResult::new();
    result.add_result(TestResult::passed("test1", 100));
    result.add_result(TestResult::failed("test2", 100, "error"));
    result.finalize();

    assert!(!result.success);
}

#[test]
fn test_result_by_name() {
    let mut result = PytestResult::new();
    result.add_result(TestResult::passed("tests/test_foo.py::test_alpha", 100));
    result.add_result(TestResult::passed("tests/test_bar.py::test_beta", 200));

    // Full match
    let found = result.by_name("tests/test_foo.py::test_alpha");
    assert!(found.is_some());

    // Partial match (ends with)
    let found = result.by_name("test_alpha");
    assert!(found.is_some());

    // Not found
    let not_found = result.by_name("test_gamma");
    assert!(not_found.is_none());
}

#[test]
fn test_result_multiple() {
    let mut result = PytestResult::new();
    result.add_result(TestResult::passed("a", 10));
    result.add_result(TestResult::passed("b", 20));
    result.add_result(TestResult::failed("c", 30, "err"));
    result.add_result(TestResult::ignored("d"));
    result.finalize();

    assert_eq!(result.total, 4);
    assert_eq!(result.passed, 2);
    assert_eq!(result.failed, 1);
    assert_eq!(result.skipped, 1);
    assert_eq!(result.total_duration_ms, 60);
    assert!(!result.success);
}

// ==================== Integration with TestResult ====================

#[test]
fn test_result_shared_types() {
    // TestResult and TestOutcome are shared with cargo runner
    let result = TestResult::passed("pytest_test", 50);
    assert_eq!(result.outcome, TestOutcome::Passed);
}
