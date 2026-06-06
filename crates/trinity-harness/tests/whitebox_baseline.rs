//! Whitebox tests for baseline recording.

use trinity_harness::graph::NodeId;
use trinity_harness::runners::{
    compare_baselines, record_baseline, Baseline, BaselineSummary, NodeState,
    StateTracker, TestFailure,
};

// ==================== Baseline ====================

#[test]
fn test_baseline_new() {
    let baseline = Baseline::new("test baseline");
    assert_eq!(baseline.description, "test baseline");
    assert!(baseline.timestamp > 0);
    assert!(baseline.node_states.is_empty());
    assert!(baseline.failures.is_empty());
}

#[test]
fn test_baseline_with_commit() {
    let baseline = Baseline::new("test").with_commit("abc123");
    assert_eq!(baseline.commit_hash, Some("abc123".to_string()));
}

#[test]
fn test_baseline_record_node() {
    let mut baseline = Baseline::new("test");
    baseline.record_node(NodeId(0), "src/lib.rs", "test_fn", NodeState::Green);

    assert_eq!(baseline.node_states.len(), 1);
    let record = baseline.node_states.get(&0).unwrap();
    assert_eq!(record.file_path, "src/lib.rs");
    assert_eq!(record.name, "test_fn");
    assert_eq!(record.state, "GREEN");
}

#[test]
fn test_baseline_record_failure() {
    let mut baseline = Baseline::new("test");
    baseline.record_failure("test_foo", Some(NodeId(0)), "assertion failed");

    assert_eq!(baseline.failures.len(), 1);
    assert_eq!(baseline.failures[0].test_name, "test_foo");
    assert!(!baseline.failures[0].triaged);
}

#[test]
fn test_baseline_compute_summary() {
    let mut baseline = Baseline::new("test");
    baseline.record_node(NodeId(0), "a.rs", "a", NodeState::Green);
    baseline.record_node(NodeId(1), "b.rs", "b", NodeState::Green);
    baseline.record_node(NodeId(2), "c.rs", "c", NodeState::Red);
    baseline.record_failure("test_c", Some(NodeId(2)), "error");
    baseline.compute_summary();

    assert_eq!(baseline.summary.total_nodes, 3);
    assert_eq!(baseline.summary.green, 2);
    assert_eq!(baseline.summary.red, 1);
    assert_eq!(baseline.summary.total_failures, 1);
}

#[test]
fn test_baseline_untriaged_failures() {
    let mut baseline = Baseline::new("test");
    baseline.record_failure("test_a", None, "error");
    baseline.record_failure("test_b", None, "error");
    baseline.triage_failure("test_a");

    let untriaged = baseline.untriaged_failures();
    assert_eq!(untriaged.len(), 1);
    assert_eq!(untriaged[0].test_name, "test_b");
}

#[test]
fn test_baseline_triage_failure() {
    let mut baseline = Baseline::new("test");
    baseline.record_failure("test_foo", None, "error");

    assert!(!baseline.failures[0].triaged);
    baseline.triage_failure("test_foo");
    assert!(baseline.failures[0].triaged);
}

#[test]
fn test_baseline_generate_report() {
    let mut baseline = Baseline::new("Initial baseline");
    baseline.record_node(NodeId(0), "a.rs", "func_a", NodeState::Green);
    baseline.record_failure("test_fail", None, "error");
    baseline.compute_summary();

    let report = baseline.generate_report();

    assert!(report.contains("Baseline Report"));
    assert!(report.contains("Initial baseline"));
    assert!(report.contains("GREEN"));
    assert!(report.contains("test_fail"));
}

// ==================== BaselineSummary ====================

#[test]
fn test_summary_default() {
    let summary = BaselineSummary::default();
    assert_eq!(summary.total_nodes, 0);
    assert_eq!(summary.health_percent, 0.0);
}

// ==================== record_baseline ====================

#[test]
fn test_record_baseline_from_tracker() {
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Red);

    let node_info = vec![
        (NodeId(0), "a.rs".to_string(), "func_a".to_string()),
        (NodeId(1), "b.rs".to_string(), "func_b".to_string()),
    ];

    let baseline = record_baseline(&tracker, "Test baseline", &node_info);

    assert_eq!(baseline.node_states.len(), 2);
    assert_eq!(baseline.summary.green, 1);
    assert_eq!(baseline.summary.red, 1);
}

// ==================== compare_baselines ====================

#[test]
fn test_compare_baselines_no_changes() {
    let mut b1 = Baseline::new("old");
    b1.record_node(NodeId(0), "a.rs", "func", NodeState::Green);
    b1.compute_summary();

    let mut b2 = Baseline::new("new");
    b2.record_node(NodeId(0), "a.rs", "func", NodeState::Green);
    b2.compute_summary();

    let comparison = compare_baselines(&b1, &b2);

    assert!(comparison.nodes_improved.is_empty());
    assert!(comparison.nodes_regressed.is_empty());
}

#[test]
fn test_compare_baselines_improvement() {
    let mut b1 = Baseline::new("old");
    b1.record_node(NodeId(0), "a.rs", "func", NodeState::Red);
    b1.compute_summary();

    let mut b2 = Baseline::new("new");
    b2.record_node(NodeId(0), "a.rs", "func", NodeState::Green);
    b2.compute_summary();

    let comparison = compare_baselines(&b1, &b2);

    assert_eq!(comparison.nodes_improved.len(), 1);
    assert!(comparison.nodes_regressed.is_empty());
}

#[test]
fn test_compare_baselines_regression() {
    let mut b1 = Baseline::new("old");
    b1.record_node(NodeId(0), "a.rs", "func", NodeState::Green);
    b1.compute_summary();

    let mut b2 = Baseline::new("new");
    b2.record_node(NodeId(0), "a.rs", "func", NodeState::Red);
    b2.compute_summary();

    let comparison = compare_baselines(&b1, &b2);

    assert!(comparison.nodes_improved.is_empty());
    assert_eq!(comparison.nodes_regressed.len(), 1);
}

#[test]
fn test_compare_baselines_new_failure() {
    let mut b1 = Baseline::new("old");
    b1.compute_summary();

    let mut b2 = Baseline::new("new");
    b2.record_failure("test_new", None, "error");
    b2.compute_summary();

    let comparison = compare_baselines(&b1, &b2);

    assert_eq!(comparison.new_failures.len(), 1);
    assert!(comparison.fixed_failures.is_empty());
}

#[test]
fn test_compare_baselines_fixed_failure() {
    let mut b1 = Baseline::new("old");
    b1.record_failure("test_old", None, "error");
    b1.compute_summary();

    let mut b2 = Baseline::new("new");
    b2.compute_summary();

    let comparison = compare_baselines(&b1, &b2);

    assert!(comparison.new_failures.is_empty());
    assert_eq!(comparison.fixed_failures.len(), 1);
}
