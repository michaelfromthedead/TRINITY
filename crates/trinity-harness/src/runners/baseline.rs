//! Baseline recording and persistence.
//!
//! Records test state snapshots with timestamps for tracking changes over time.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{NodeState, StateTracker};
use crate::graph::NodeId;

/// A recorded baseline snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    /// Timestamp when baseline was recorded (Unix epoch seconds).
    pub timestamp: u64,
    /// Human-readable timestamp.
    pub timestamp_str: String,
    /// Git commit hash if available.
    pub commit_hash: Option<String>,
    /// Description/reason for this baseline.
    pub description: String,
    /// State of each node at baseline time.
    pub node_states: HashMap<usize, NodeStateRecord>,
    /// Test failures at baseline time.
    pub failures: Vec<TestFailure>,
    /// Summary statistics.
    pub summary: BaselineSummary,
}

/// Recorded state of a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStateRecord {
    /// Node ID.
    pub node_id: usize,
    /// File path.
    pub file_path: String,
    /// Node name.
    pub name: String,
    /// State at baseline time.
    pub state: String,
}

/// A test failure for triage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFailure {
    /// Test name.
    pub test_name: String,
    /// Target node ID.
    pub node_id: Option<usize>,
    /// Failure message.
    pub message: String,
    /// Whether it's been triaged.
    pub triaged: bool,
}

/// Summary of baseline state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaselineSummary {
    /// Total nodes.
    pub total_nodes: usize,
    /// Nodes in GREEN state.
    pub green: usize,
    /// Nodes in RED state.
    pub red: usize,
    /// Nodes in DIRTY state.
    pub dirty: usize,
    /// Nodes in UNTESTED state.
    pub untested: usize,
    /// Total failures.
    pub total_failures: usize,
    /// Health percentage.
    pub health_percent: f64,
}

impl Baseline {
    /// Create a new baseline from current state.
    pub fn new(description: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            timestamp: now,
            timestamp_str: format_timestamp(now),
            commit_hash: None,
            description: description.into(),
            node_states: HashMap::new(),
            failures: Vec::new(),
            summary: BaselineSummary::default(),
        }
    }

    /// Set the git commit hash.
    pub fn with_commit(mut self, hash: impl Into<String>) -> Self {
        self.commit_hash = Some(hash.into());
        self
    }

    /// Record a node state.
    pub fn record_node(&mut self, node_id: NodeId, file_path: &str, name: &str, state: NodeState) {
        self.node_states.insert(
            node_id.0,
            NodeStateRecord {
                node_id: node_id.0,
                file_path: file_path.to_string(),
                name: name.to_string(),
                state: format!("{}", state),
            },
        );
    }

    /// Record a test failure.
    pub fn record_failure(&mut self, test_name: &str, node_id: Option<NodeId>, message: &str) {
        self.failures.push(TestFailure {
            test_name: test_name.to_string(),
            node_id: node_id.map(|id| id.0),
            message: message.to_string(),
            triaged: false,
        });
    }

    /// Compute summary statistics.
    pub fn compute_summary(&mut self) {
        let mut summary = BaselineSummary::default();
        summary.total_nodes = self.node_states.len();

        for record in self.node_states.values() {
            match record.state.as_str() {
                "GREEN" => summary.green += 1,
                "RED" => summary.red += 1,
                "DIRTY" => summary.dirty += 1,
                "UNTESTED" => summary.untested += 1,
                _ => {}
            }
        }

        summary.total_failures = self.failures.len();
        summary.health_percent = if summary.total_nodes > 0 {
            (summary.green as f64 / summary.total_nodes as f64) * 100.0
        } else {
            0.0
        };

        self.summary = summary;
    }

    /// Get untriaged failures.
    pub fn untriaged_failures(&self) -> Vec<&TestFailure> {
        self.failures.iter().filter(|f| !f.triaged).collect()
    }

    /// Mark a failure as triaged.
    pub fn triage_failure(&mut self, test_name: &str) {
        for failure in &mut self.failures {
            if failure.test_name == test_name {
                failure.triaged = true;
            }
        }
    }

    /// Generate a report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Baseline Report ===\n\n");
        report.push_str(&format!("Timestamp: {}\n", self.timestamp_str));
        if let Some(ref hash) = self.commit_hash {
            report.push_str(&format!("Commit: {}\n", hash));
        }
        report.push_str(&format!("Description: {}\n\n", self.description));

        report.push_str("Summary:\n");
        report.push_str(&format!("  Total nodes: {}\n", self.summary.total_nodes));
        report.push_str(&format!("  GREEN:       {}\n", self.summary.green));
        report.push_str(&format!("  RED:         {}\n", self.summary.red));
        report.push_str(&format!("  DIRTY:       {}\n", self.summary.dirty));
        report.push_str(&format!("  UNTESTED:    {}\n", self.summary.untested));
        report.push_str(&format!("  Health:      {:.1}%\n", self.summary.health_percent));

        if !self.failures.is_empty() {
            report.push_str(&format!("\nFailures ({}):\n", self.failures.len()));
            for failure in &self.failures {
                let status = if failure.triaged { "✓" } else { "○" };
                report.push_str(&format!("  {} {}\n", status, failure.test_name));
            }
        }

        report
    }

    /// Save baseline to a JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load baseline from a JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

/// Record a baseline from a state tracker.
pub fn record_baseline(
    tracker: &StateTracker,
    description: &str,
    node_info: &[(NodeId, String, String)], // (id, file_path, name)
) -> Baseline {
    let mut baseline = Baseline::new(description);

    for (node_id, file_path, name) in node_info {
        let state = tracker.get_state(*node_id);
        baseline.record_node(*node_id, file_path, name, state);
    }

    baseline.compute_summary();
    baseline
}

/// Compare two baselines.
pub fn compare_baselines(old: &Baseline, new: &Baseline) -> BaselineComparison {
    let mut comparison = BaselineComparison {
        old_timestamp: old.timestamp_str.clone(),
        new_timestamp: new.timestamp_str.clone(),
        nodes_improved: Vec::new(),
        nodes_regressed: Vec::new(),
        new_failures: Vec::new(),
        fixed_failures: Vec::new(),
    };

    // Compare node states
    for (id, new_record) in &new.node_states {
        if let Some(old_record) = old.node_states.get(id) {
            if old_record.state != new_record.state {
                let change = StateChange {
                    node_id: *id,
                    name: new_record.name.clone(),
                    old_state: old_record.state.clone(),
                    new_state: new_record.state.clone(),
                };

                if is_improvement(&old_record.state, &new_record.state) {
                    comparison.nodes_improved.push(change);
                } else {
                    comparison.nodes_regressed.push(change);
                }
            }
        }
    }

    // Compare failures
    let old_tests: std::collections::HashSet<_> = old.failures.iter().map(|f| &f.test_name).collect();
    let new_tests: std::collections::HashSet<_> = new.failures.iter().map(|f| &f.test_name).collect();

    for test in new_tests.difference(&old_tests) {
        comparison.new_failures.push((*test).clone());
    }

    for test in old_tests.difference(&new_tests) {
        comparison.fixed_failures.push((*test).clone());
    }

    comparison
}

/// Comparison between two baselines.
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    /// Old baseline timestamp.
    pub old_timestamp: String,
    /// New baseline timestamp.
    pub new_timestamp: String,
    /// Nodes that improved (e.g., RED -> GREEN).
    pub nodes_improved: Vec<StateChange>,
    /// Nodes that regressed (e.g., GREEN -> RED).
    pub nodes_regressed: Vec<StateChange>,
    /// New test failures.
    pub new_failures: Vec<String>,
    /// Fixed test failures.
    pub fixed_failures: Vec<String>,
}

/// A state change for a node.
#[derive(Debug, Clone)]
pub struct StateChange {
    /// Node ID.
    pub node_id: usize,
    /// Node name.
    pub name: String,
    /// Old state.
    pub old_state: String,
    /// New state.
    pub new_state: String,
}

/// Check if a state transition is an improvement.
fn is_improvement(old: &str, new: &str) -> bool {
    matches!(
        (old, new),
        ("RED", "GREEN") | ("UNTESTED", "GREEN") | ("DIRTY", "GREEN") | ("RED", "DIRTY")
    )
}

/// Format a Unix timestamp as a human-readable string.
fn format_timestamp(secs: u64) -> String {
    // Simple formatting without external crates
    format!("{}", secs)
}
