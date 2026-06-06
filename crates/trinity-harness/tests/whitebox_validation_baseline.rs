//! Whitebox tests for baseline validation.

use trinity_harness::graph::NodeId;
use trinity_harness::runners::{
    generate_summary, validate_and_summarize, validate_baseline, validate_tracker,
    Baseline, NodeState, StateTracker, ValidationResult,
};

// ==================== ValidationResult ====================

#[test]
fn test_validation_result_new() {
    let result = ValidationResult::new();
    assert!(!result.is_valid);
    assert_eq!(result.total_nodes, 0);
    assert!(result.errors.is_empty());
}

#[test]
fn test_validation_result_passed() {
    let mut result = ValidationResult::new();
    result.is_valid = true;

    assert!(result.passed());
}

#[test]
fn test_validation_result_passed_with_errors() {
    let mut result = ValidationResult::new();
    result.is_valid = true;
    result.errors.push("error".to_string());

    assert!(!result.passed());
}

#[test]
fn test_validation_result_health_percent() {
    let mut result = ValidationResult::new();
    result.total_nodes = 10;
    result.green_count = 8;

    assert_eq!(result.health_percent(), 80.0);
}

#[test]
fn test_validation_result_health_percent_zero() {
    let result = ValidationResult::new();
    assert_eq!(result.health_percent(), 0.0);
}

#[test]
fn test_validation_result_all_states_known() {
    let mut result = ValidationResult::new();
    result.unknown_count = 0;

    assert!(result.all_states_known());
}

#[test]
fn test_validation_result_generate_report() {
    let mut result = ValidationResult::new();
    result.is_valid = true;
    result.total_nodes = 10;
    result.green_count = 8;
    result.red_count = 2;
    result.warnings.push("2 nodes have failing tests".to_string());

    let report = result.generate_report();

    assert!(report.contains("Validation Report"));
    assert!(report.contains("PASSED"));
    assert!(report.contains("GREEN"));
    assert!(report.contains("RED"));
    assert!(report.contains("80.0%"));
}

#[test]
fn test_validation_result_generate_report_failed() {
    let mut result = ValidationResult::new();
    result.is_valid = false;
    result.errors.push("Unknown state".to_string());

    let report = result.generate_report();

    assert!(report.contains("FAILED"));
    assert!(report.contains("Unknown state"));
}

// ==================== validate_baseline ====================

#[test]
fn test_validate_baseline_valid() {
    let mut baseline = Baseline::new("test");
    baseline.record_node(NodeId(0), "a.rs", "func_a", NodeState::Green);
    baseline.record_node(NodeId(1), "b.rs", "func_b", NodeState::Red);
    baseline.compute_summary();

    let result = validate_baseline(&baseline);

    assert!(result.is_valid);
    assert_eq!(result.green_count, 1);
    assert_eq!(result.red_count, 1);
    assert_eq!(result.unknown_count, 0);
}

#[test]
fn test_validate_baseline_with_untested() {
    let mut baseline = Baseline::new("test");
    baseline.record_node(NodeId(0), "a.rs", "func", NodeState::Untested);
    baseline.compute_summary();

    let result = validate_baseline(&baseline);

    assert!(result.is_valid);
    assert_eq!(result.untested_count, 1);
    assert!(!result.warnings.is_empty());
}

#[test]
fn test_validate_baseline_empty() {
    let baseline = Baseline::new("test");
    let result = validate_baseline(&baseline);

    assert!(result.is_valid);
    assert_eq!(result.total_nodes, 0);
}

// ==================== validate_tracker ====================

#[test]
fn test_validate_tracker() {
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Green);
    tracker.set_state(NodeId(2), NodeState::Red);

    let node_ids = vec![NodeId(0), NodeId(1), NodeId(2)];
    let result = validate_tracker(&tracker, &node_ids);

    assert!(result.is_valid);
    assert_eq!(result.green_count, 2);
    assert_eq!(result.red_count, 1);
}

#[test]
fn test_validate_tracker_with_untested() {
    let tracker = StateTracker::new();
    let node_ids = vec![NodeId(0), NodeId(1)];

    let result = validate_tracker(&tracker, &node_ids);

    assert!(result.is_valid);
    assert_eq!(result.untested_count, 2);
}

// ==================== generate_summary ====================

#[test]
fn test_generate_summary() {
    let mut baseline = Baseline::new("Test baseline");
    baseline.record_node(NodeId(0), "a.rs", "func", NodeState::Green);
    baseline.compute_summary();

    let summary = generate_summary(&baseline);

    assert!(summary.contains("Baseline Summary"));
    assert!(summary.contains("Test baseline"));
    assert!(summary.contains("GREEN"));
}

#[test]
fn test_generate_summary_with_failures() {
    let mut baseline = Baseline::new("Test");
    baseline.record_failure("test_fail", None, "error");
    baseline.compute_summary();

    let summary = generate_summary(&baseline);

    assert!(summary.contains("Failures"));
}

// ==================== validate_and_summarize ====================

#[test]
fn test_validate_and_summarize() {
    let mut baseline = Baseline::new("Test");
    baseline.record_node(NodeId(0), "a.rs", "func", NodeState::Green);
    baseline.compute_summary();

    let (result, summary) = validate_and_summarize(&baseline);

    assert!(result.is_valid);
    assert!(summary.contains("Baseline Summary"));
}
