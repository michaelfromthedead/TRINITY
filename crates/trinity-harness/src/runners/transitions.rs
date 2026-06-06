//! State transitions based on test results.
//!
//! Updates code node states based on test execution outcomes.

use std::collections::HashMap;

use super::MappingResult;
use crate::graph::NodeId;

/// State of a code node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeState {
    /// Node has not been tested.
    Untested,
    /// All tests for this node passed.
    Green,
    /// Some tests for this node failed.
    Red,
    /// Node has been modified since last test.
    Dirty,
}

impl Default for NodeState {
    fn default() -> Self {
        Self::Untested
    }
}

impl std::fmt::Display for NodeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Untested => write!(f, "UNTESTED"),
            Self::Green => write!(f, "GREEN"),
            Self::Red => write!(f, "RED"),
            Self::Dirty => write!(f, "DIRTY"),
        }
    }
}

/// Test event that triggers a state transition.
#[derive(Debug, Clone)]
pub enum TestEvent {
    /// All tests for a node passed.
    TestsPassed { node_id: NodeId },
    /// Some tests for a node failed.
    TestsFailed { node_id: NodeId, failed_tests: Vec<String> },
    /// Node code was modified.
    CodeModified { node_id: NodeId },
    /// Reset node to untested.
    Reset { node_id: NodeId },
}

/// Tracks state for all code nodes.
#[derive(Debug, Clone, Default)]
pub struct StateTracker {
    /// State of each node.
    states: HashMap<NodeId, NodeState>,
    /// History of state transitions.
    history: Vec<StateTransition>,
}

/// A recorded state transition.
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// Node that transitioned.
    pub node_id: NodeId,
    /// Previous state.
    pub from: NodeState,
    /// New state.
    pub to: NodeState,
    /// Event that caused the transition.
    pub event: String,
}

impl StateTracker {
    /// Create a new state tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the state of a node.
    pub fn get_state(&self, node_id: NodeId) -> NodeState {
        self.states.get(&node_id).copied().unwrap_or_default()
    }

    /// Set the state of a node directly.
    pub fn set_state(&mut self, node_id: NodeId, state: NodeState) {
        let from = self.get_state(node_id);
        self.states.insert(node_id, state);
        self.history.push(StateTransition {
            node_id,
            from,
            to: state,
            event: "set".to_string(),
        });
    }

    /// Handle a test event and update states.
    pub fn handle_event(&mut self, event: TestEvent) {
        match event {
            TestEvent::TestsPassed { node_id } => {
                self.transition(node_id, NodeState::Green, "tests_passed");
            }
            TestEvent::TestsFailed { node_id, .. } => {
                self.transition(node_id, NodeState::Red, "tests_failed");
            }
            TestEvent::CodeModified { node_id } => {
                let current = self.get_state(node_id);
                if current == NodeState::Green {
                    self.transition(node_id, NodeState::Dirty, "code_modified");
                }
            }
            TestEvent::Reset { node_id } => {
                self.transition(node_id, NodeState::Untested, "reset");
            }
        }
    }

    /// Perform a state transition.
    fn transition(&mut self, node_id: NodeId, to: NodeState, event: &str) {
        let from = self.get_state(node_id);
        if from != to {
            self.states.insert(node_id, to);
            self.history.push(StateTransition {
                node_id,
                from,
                to,
                event: event.to_string(),
            });
        }
    }

    /// Apply test results to update node states.
    pub fn apply_results(&mut self, results: &MappingResult) {
        for (&node_id, result) in &results.by_node {
            if result.has_failures() {
                self.handle_event(TestEvent::TestsFailed {
                    node_id,
                    failed_tests: result.failed_tests.clone(),
                });
            } else if result.all_passed() {
                self.handle_event(TestEvent::TestsPassed { node_id });
            }
        }
    }

    /// Get all nodes in a specific state.
    pub fn nodes_in_state(&self, state: NodeState) -> Vec<NodeId> {
        self.states
            .iter()
            .filter(|(_, &s)| s == state)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get state counts.
    pub fn state_counts(&self) -> HashMap<NodeState, usize> {
        let mut counts = HashMap::new();
        for &state in self.states.values() {
            *counts.entry(state).or_insert(0) += 1;
        }
        counts
    }

    /// Get transition history.
    pub fn history(&self) -> &[StateTransition] {
        &self.history
    }

    /// Get summary stats.
    pub fn summary(&self) -> StateSummary {
        let counts = self.state_counts();
        StateSummary {
            green: *counts.get(&NodeState::Green).unwrap_or(&0),
            red: *counts.get(&NodeState::Red).unwrap_or(&0),
            dirty: *counts.get(&NodeState::Dirty).unwrap_or(&0),
            untested: *counts.get(&NodeState::Untested).unwrap_or(&0),
            total: self.states.len(),
        }
    }

    /// Generate a report.
    pub fn generate_report(&self) -> String {
        let summary = self.summary();
        let mut report = String::new();

        report.push_str("=== State Summary ===\n");
        report.push_str(&format!("Total nodes: {}\n", summary.total));
        report.push_str(&format!("GREEN:    {} nodes\n", summary.green));
        report.push_str(&format!("RED:      {} nodes\n", summary.red));
        report.push_str(&format!("DIRTY:    {} nodes\n", summary.dirty));
        report.push_str(&format!("UNTESTED: {} nodes\n", summary.untested));

        if summary.total > 0 {
            let health = (summary.green as f64 / summary.total as f64) * 100.0;
            report.push_str(&format!("\nHealth: {:.1}%\n", health));
        }

        report
    }
}

/// Summary of node states.
#[derive(Debug, Clone, Default)]
pub struct StateSummary {
    /// Nodes with all tests passing.
    pub green: usize,
    /// Nodes with failing tests.
    pub red: usize,
    /// Nodes modified since last test.
    pub dirty: usize,
    /// Nodes not yet tested.
    pub untested: usize,
    /// Total tracked nodes.
    pub total: usize,
}

impl StateSummary {
    /// Get health percentage (green / total).
    pub fn health_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.green as f64 / self.total as f64) * 100.0
        }
    }

    /// Check if all nodes are green.
    pub fn all_green(&self) -> bool {
        self.red == 0 && self.dirty == 0 && self.green > 0
    }

    /// Check if any nodes are red.
    pub fn has_failures(&self) -> bool {
        self.red > 0
    }
}
