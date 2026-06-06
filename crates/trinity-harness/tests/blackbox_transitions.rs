//! Blackbox tests for state transitions with mapping results.

use std::collections::HashMap;
use trinity_harness::graph::NodeId;
use trinity_harness::runners::{
    map_results, MappingResult, NodeResult, NodeState, StateTracker, TestResult,
};

#[test]
fn test_apply_results_all_passed() {
    let mut tracker = StateTracker::new();
    
    let mut result = MappingResult::new();
    let mut nr = NodeResult::new(NodeId(0));
    nr.total = 3;
    nr.passed = 3;
    result.by_node.insert(NodeId(0), nr);

    tracker.apply_results(&result);

    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Green);
}

#[test]
fn test_apply_results_with_failures() {
    let mut tracker = StateTracker::new();
    
    let mut result = MappingResult::new();
    let mut nr = NodeResult::new(NodeId(0));
    nr.total = 3;
    nr.passed = 2;
    nr.failed = 1;
    nr.failed_tests = vec!["test_foo".to_string()];
    result.by_node.insert(NodeId(0), nr);

    tracker.apply_results(&result);

    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Red);
}

#[test]
fn test_apply_results_mixed() {
    let mut tracker = StateTracker::new();
    
    let mut result = MappingResult::new();
    
    // Node 0: all passed
    let mut nr0 = NodeResult::new(NodeId(0));
    nr0.total = 2;
    nr0.passed = 2;
    result.by_node.insert(NodeId(0), nr0);

    // Node 1: failures
    let mut nr1 = NodeResult::new(NodeId(1));
    nr1.total = 2;
    nr1.passed = 1;
    nr1.failed = 1;
    result.by_node.insert(NodeId(1), nr1);

    tracker.apply_results(&result);

    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Green);
    assert_eq!(tracker.get_state(NodeId(1)), NodeState::Red);
}

#[test]
fn test_state_workflow() {
    let mut tracker = StateTracker::new();

    // Initial: untested
    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Untested);

    // Run tests - pass
    let mut result = MappingResult::new();
    let mut nr = NodeResult::new(NodeId(0));
    nr.total = 1;
    nr.passed = 1;
    result.by_node.insert(NodeId(0), nr);
    tracker.apply_results(&result);

    // Now green
    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Green);

    // Modify code
    tracker.handle_event(trinity_harness::runners::TestEvent::CodeModified { 
        node_id: NodeId(0) 
    });

    // Now dirty
    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Dirty);

    // Run tests again - pass
    tracker.apply_results(&result);

    // Green again
    assert_eq!(tracker.get_state(NodeId(0)), NodeState::Green);
}

#[test]
fn test_summary_after_apply() {
    let mut tracker = StateTracker::new();
    
    let mut result = MappingResult::new();
    
    for i in 0..10 {
        let mut nr = NodeResult::new(NodeId(i));
        nr.total = 1;
        if i < 8 {
            nr.passed = 1;
        } else {
            nr.failed = 1;
        }
        result.by_node.insert(NodeId(i), nr);
    }

    tracker.apply_results(&result);

    let summary = tracker.summary();
    assert_eq!(summary.green, 8);
    assert_eq!(summary.red, 2);
    assert_eq!(summary.health_percent(), 80.0);
}
