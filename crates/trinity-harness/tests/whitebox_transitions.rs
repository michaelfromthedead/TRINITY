//! Whitebox tests for state transitions.

use trinity_harness::graph::NodeId;
use trinity_harness::runners::{
    NodeState, StateTracker, StateSummary, TestEvent,
};

// ==================== NodeState ====================

#[test]
fn test_node_state_default() {
    let state = NodeState::default();
    assert_eq!(state, NodeState::Untested);
}

#[test]
fn test_node_state_display() {
    assert_eq!(format!("{}", NodeState::Green), "GREEN");
    assert_eq!(format!("{}", NodeState::Red), "RED");
    assert_eq!(format!("{}", NodeState::Dirty), "DIRTY");
    assert_eq!(format!("{}", NodeState::Untested), "UNTESTED");
}

// ==================== StateTracker ====================

#[test]
fn test_tracker_new() {
    let tracker = StateTracker::new();
    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Untested);
}

#[test]
fn test_tracker_set_state() {
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);

    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Green);
}

#[test]
fn test_tracker_tests_passed() {
    let mut tracker = StateTracker::new();
    tracker.handle_event(TestEvent::TestsPassed { node_id: NodeId(0) });

    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Green);
}

#[test]
fn test_tracker_tests_failed() {
    let mut tracker = StateTracker::new();
    tracker.handle_event(TestEvent::TestsFailed {
        node_id: NodeId(0),
        failed_tests: vec!["test_foo".to_string()],
    });

    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Red);
}

#[test]
fn test_tracker_code_modified() {
    let mut tracker = StateTracker::new();
    
    // First make it green
    tracker.set_state(NodeId(0), NodeState::Green);
    
    // Then modify code
    tracker.handle_event(TestEvent::CodeModified { node_id: NodeId(0) });

    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Dirty);
}

#[test]
fn test_tracker_code_modified_not_green() {
    let mut tracker = StateTracker::new();
    
    // If not green, code modified doesn't change state
    tracker.set_state(NodeId(0), NodeState::Red);
    tracker.handle_event(TestEvent::CodeModified { node_id: NodeId(0) });

    // Still red
    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Red);
}

#[test]
fn test_tracker_reset() {
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.handle_event(TestEvent::Reset { node_id: NodeId(0) });

    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Untested);
}

#[test]
fn test_tracker_nodes_in_state() {
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Green);
    tracker.set_state(NodeId(2), NodeState::Red);

    let greens = tracker.nodes_in_state(NodeState::Green);
    assert_eq!(greens.len(), 2);

    let reds = tracker.nodes_in_state(NodeState::Red);
    assert_eq!(reds.len(), 1);
}

#[test]
fn test_tracker_state_counts() {
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Green);
    tracker.set_state(NodeId(2), NodeState::Red);

    let counts = tracker.state_counts();
    assert_eq!(*counts.get(&NodeState::Green).unwrap_or(&0), 2);
    assert_eq!(*counts.get(&NodeState::Red).unwrap_or(&0), 1);
}

#[test]
fn test_tracker_history() {
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.handle_event(TestEvent::CodeModified { node_id: NodeId(0) });

    let history = tracker.history();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].to, NodeState::Green);
    assert_eq!(history[1].to, NodeState::Dirty);
}

#[test]
fn test_tracker_summary() {
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Green);
    tracker.set_state(NodeId(2), NodeState::Red);
    tracker.set_state(NodeId(3), NodeState::Dirty);

    let summary = tracker.summary();
    assert_eq!(summary.green, 2);
    assert_eq!(summary.red, 1);
    assert_eq!(summary.dirty, 1);
    assert_eq!(summary.total, 4);
}

#[test]
fn test_tracker_generate_report() {
    let mut tracker = StateTracker::new();
    tracker.set_state(NodeId(0), NodeState::Green);
    tracker.set_state(NodeId(1), NodeState::Red);

    let report = tracker.generate_report();
    assert!(report.contains("State Summary"));
    assert!(report.contains("GREEN"));
    assert!(report.contains("RED"));
}

// ==================== StateSummary ====================

#[test]
fn test_summary_health_percent() {
    let summary = StateSummary {
        green: 8,
        red: 2,
        dirty: 0,
        untested: 0,
        total: 10,
    };

    assert_eq!(summary.health_percent(), 80.0);
}

#[test]
fn test_summary_health_percent_zero() {
    let summary = StateSummary::default();
    assert_eq!(summary.health_percent(), 0.0);
}

#[test]
fn test_summary_all_green() {
    let summary = StateSummary {
        green: 10,
        red: 0,
        dirty: 0,
        untested: 0,
        total: 10,
    };

    assert!(summary.all_green());
}

#[test]
fn test_summary_has_failures() {
    let summary = StateSummary {
        green: 8,
        red: 2,
        dirty: 0,
        untested: 0,
        total: 10,
    };

    assert!(summary.has_failures());
}
