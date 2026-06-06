//! Baseline validation and reporting.
//!
//! Validates that all nodes have proper states and generates summary reports.

use super::{Baseline, NodeState, StateTracker};
use crate::graph::NodeId;

/// Result of validating a baseline.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// Whether the baseline is valid.
    pub is_valid: bool,
    /// Total nodes checked.
    pub total_nodes: usize,
    /// Nodes in GREEN state.
    pub green_count: usize,
    /// Nodes in RED state.
    pub red_count: usize,
    /// Nodes in UNTESTED state.
    pub untested_count: usize,
    /// Nodes in DIRTY state.
    pub dirty_count: usize,
    /// Nodes with unknown/invalid state.
    pub unknown_count: usize,
    /// Validation errors.
    pub errors: Vec<String>,
    /// Warnings (non-blocking issues).
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if validation passed.
    pub fn passed(&self) -> bool {
        self.is_valid && self.errors.is_empty()
    }

    /// Get health percentage (green / total).
    pub fn health_percent(&self) -> f64 {
        if self.total_nodes == 0 {
            0.0
        } else {
            (self.green_count as f64 / self.total_nodes as f64) * 100.0
        }
    }

    /// Check if all nodes have known states.
    pub fn all_states_known(&self) -> bool {
        self.unknown_count == 0
    }

    /// Generate validation report.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Baseline Validation Report ===\n\n");

        let status = if self.passed() { "PASSED ✓" } else { "FAILED ✗" };
        report.push_str(&format!("Status: {}\n\n", status));

        report.push_str("State Distribution:\n");
        report.push_str(&format!("  Total:    {}\n", self.total_nodes));
        report.push_str(&format!("  GREEN:    {} ({:.1}%)\n", 
            self.green_count, 
            self.percent(self.green_count)));
        report.push_str(&format!("  RED:      {} ({:.1}%)\n", 
            self.red_count,
            self.percent(self.red_count)));
        report.push_str(&format!("  UNTESTED: {} ({:.1}%)\n", 
            self.untested_count,
            self.percent(self.untested_count)));
        report.push_str(&format!("  DIRTY:    {} ({:.1}%)\n", 
            self.dirty_count,
            self.percent(self.dirty_count)));
        report.push_str(&format!("  UNKNOWN:  {}\n", self.unknown_count));

        report.push_str(&format!("\nHealth: {:.1}%\n", self.health_percent()));

        if !self.errors.is_empty() {
            report.push_str("\nErrors:\n");
            for err in &self.errors {
                report.push_str(&format!("  ✗ {}\n", err));
            }
        }

        if !self.warnings.is_empty() {
            report.push_str("\nWarnings:\n");
            for warn in &self.warnings {
                report.push_str(&format!("  ⚠ {}\n", warn));
            }
        }

        report
    }

    /// Calculate percentage.
    fn percent(&self, count: usize) -> f64 {
        if self.total_nodes == 0 {
            0.0
        } else {
            (count as f64 / self.total_nodes as f64) * 100.0
        }
    }
}

/// Validate a baseline.
pub fn validate_baseline(baseline: &Baseline) -> ValidationResult {
    let mut result = ValidationResult::new();

    result.total_nodes = baseline.node_states.len();

    // Count states
    for record in baseline.node_states.values() {
        match record.state.as_str() {
            "GREEN" => result.green_count += 1,
            "RED" => result.red_count += 1,
            "UNTESTED" => result.untested_count += 1,
            "DIRTY" => result.dirty_count += 1,
            _ => {
                result.unknown_count += 1;
                result.errors.push(format!(
                    "Node {} has unknown state: {}", 
                    record.name, 
                    record.state
                ));
            }
        }
    }

    // Check for issues
    if result.unknown_count > 0 {
        result.is_valid = false;
    } else {
        result.is_valid = true;
    }

    // Warnings
    if result.untested_count > 0 {
        let pct = result.percent(result.untested_count);
        result.warnings.push(format!(
            "{} nodes ({:.1}%) are untested", 
            result.untested_count, 
            pct
        ));
    }

    if result.red_count > 0 {
        result.warnings.push(format!(
            "{} nodes have failing tests", 
            result.red_count
        ));
    }

    result
}

/// Validate a state tracker directly.
pub fn validate_tracker(
    tracker: &StateTracker,
    node_ids: &[NodeId],
) -> ValidationResult {
    let mut result = ValidationResult::new();

    result.total_nodes = node_ids.len();

    for &node_id in node_ids {
        let state = tracker.get_state(node_id);
        match state {
            NodeState::Green => result.green_count += 1,
            NodeState::Red => result.red_count += 1,
            NodeState::Untested => result.untested_count += 1,
            NodeState::Dirty => result.dirty_count += 1,
        }
    }

    // All states are known (no unknown variant)
    result.is_valid = true;

    // Warnings
    if result.untested_count > 0 {
        result.warnings.push(format!(
            "{} nodes are untested", 
            result.untested_count
        ));
    }

    if result.red_count > 0 {
        result.warnings.push(format!(
            "{} nodes have failing tests", 
            result.red_count
        ));
    }

    result
}

/// Generate a summary from baseline stats.
pub fn generate_summary(baseline: &Baseline) -> String {
    let mut summary = String::new();

    summary.push_str("=== Baseline Summary ===\n\n");
    summary.push_str(&format!("Timestamp: {}\n", baseline.timestamp_str));
    summary.push_str(&format!("Description: {}\n\n", baseline.description));

    let s = &baseline.summary;
    summary.push_str("State Counts:\n");
    summary.push_str(&format!("  GREEN:    {}\n", s.green));
    summary.push_str(&format!("  RED:      {}\n", s.red));
    summary.push_str(&format!("  DIRTY:    {}\n", s.dirty));
    summary.push_str(&format!("  UNTESTED: {}\n", s.untested));
    summary.push_str(&format!("  Total:    {}\n", s.total_nodes));

    summary.push_str(&format!("\nHealth: {:.1}%\n", s.health_percent));

    if s.total_nodes > 0 {
        let tested_pct = ((s.green + s.red + s.dirty) as f64 / s.total_nodes as f64) * 100.0;
        summary.push_str(&format!("Test Coverage: {:.1}%\n", tested_pct));
    }

    if baseline.failures.len() > 0 {
        summary.push_str(&format!("\nFailures: {}\n", baseline.failures.len()));
        let untriaged = baseline.untriaged_failures().len();
        if untriaged > 0 {
            summary.push_str(&format!("  Untriaged: {}\n", untriaged));
        }
    }

    summary
}

/// Validate and summarize in one call.
pub fn validate_and_summarize(baseline: &Baseline) -> (ValidationResult, String) {
    let validation = validate_baseline(baseline);
    let summary = generate_summary(baseline);
    (validation, summary)
}
