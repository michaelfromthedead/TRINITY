//! Blackbox tests for baseline validation with full workflow.

use trinity_harness::graph::NodeId;
use trinity_harness::runners::{
    record_baseline, validate_and_summarize, validate_baseline, Baseline, NodeState,
    StateTracker,
};

#[test]
fn test_full_validation_workflow() {
    // Create a tracker with some states
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Green);
    tracker.set_state(NodeId(2), NodeState::Red);
    tracker.set_state(NodeId(3), NodeState::Untested);

    // Record baseline
    let node_info = vec![
        (NodeId(0), "src/a.rs".to_string(), "func_a".to_string()),
        (NodeId(1), "src/b.rs".to_string(), "func_b".to_string()),
        (NodeId(2), "src/c.rs".to_string(), "func_c".to_string()),
        (NodeId(3), "src/d.rs".to_string(), "func_d".to_string()),
    ];

    let baseline = record_baseline(&tracker, "Full workflow test", &node_info);

    // Validate
    let result = validate_baseline(&baseline);

    assert!(result.is_valid);
    assert_eq!(result.total_nodes, 4);
    assert_eq!(result.green_count, 2);
    assert_eq!(result.red_count, 1);
    assert_eq!(result.untested_count, 1);
    assert_eq!(result.health_percent(), 50.0);
}

#[test]
fn test_validate_and_summarize_workflow() {
    let mut baseline = Baseline::new("Integration test");
    baseline.record_node(NodeId(0), "a.rs", "func_a", NodeState::Green);
    baseline.record_node(NodeId(1), "b.rs", "func_b", NodeState::Green);
    baseline.record_node(NodeId(2), "c.rs", "func_c", NodeState::Green);
    baseline.record_node(NodeId(3), "d.rs", "func_d", NodeState::Red);
    baseline.record_failure("test_d", Some(NodeId(3)), "assertion failed");
    baseline.compute_summary();

    let (result, summary) = validate_and_summarize(&baseline);

    // Validation
    assert!(result.is_valid);
    assert_eq!(result.green_count, 3);
    assert_eq!(result.red_count, 1);
    assert_eq!(result.health_percent(), 75.0);

    // Summary
    assert!(summary.contains("Baseline Summary"));
    assert!(summary.contains("GREEN"));
    assert!(summary.contains("Failures"));
}

#[test]
fn test_validation_report_format() {
    let mut baseline = Baseline::new("Report format test");
    baseline.record_node(NodeId(0), "a.rs", "func", NodeState::Green);
    baseline.compute_summary();

    let result = validate_baseline(&baseline);
    let report = result.generate_report();

    // Check report structure
    assert!(report.contains("Baseline Validation Report"));
    assert!(report.contains("Status:"));
    assert!(report.contains("State Distribution:"));
    assert!(report.contains("Health:"));
}

#[test]
fn test_validation_all_green() {
    let mut baseline = Baseline::new("All green");
    for i in 0..10 {
        baseline.record_node(NodeId(i), &format!("{}.rs", i), &format!("func_{}", i), NodeState::Green);
    }
    baseline.compute_summary();

    let result = validate_baseline(&baseline);

    assert!(result.is_valid);
    assert!(result.passed());
    assert_eq!(result.health_percent(), 100.0);
    assert!(result.warnings.is_empty());
}

#[test]
fn test_validation_all_untested() {
    let mut baseline = Baseline::new("All untested");
    for i in 0..5 {
        baseline.record_node(NodeId(i), &format!("{}.rs", i), &format!("func_{}", i), NodeState::Untested);
    }
    baseline.compute_summary();

    let result = validate_baseline(&baseline);

    assert!(result.is_valid); // Untested is a valid state
    assert_eq!(result.untested_count, 5);
    assert_eq!(result.health_percent(), 0.0);
    assert!(!result.warnings.is_empty()); // Should warn about untested
}
