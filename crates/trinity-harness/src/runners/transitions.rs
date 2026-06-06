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

// =============================================================================
// Database-backed State Tracker
// =============================================================================

use crate::db::HarnessDb;

/// Database-backed state tracker that persists state to SQLite.
pub struct DbStateTracker<'a> {
    db: &'a HarnessDb,
}

impl<'a> DbStateTracker<'a> {
    /// Create a new database-backed state tracker.
    pub fn new(db: &'a HarnessDb) -> Self {
        Self { db }
    }

    /// Get state of a node by its database ID (file:line:name format).
    pub fn get_state_by_id(&self, node_id: &str) -> NodeState {
        let conn = self.db.connection();
        let result: Result<String, _> = conn.query_row(
            "SELECT current_state FROM code_nodes WHERE node_id = ?1",
            [node_id],
            |row| row.get(0),
        );

        match result {
            Ok(state_str) => db_state_to_enum(&state_str),
            Err(_) => NodeState::Untested,
        }
    }

    /// Set state of a node by its database ID.
    pub fn set_state_by_id(&self, node_id: &str, state: NodeState) -> Result<(), String> {
        let conn = self.db.connection();
        let state_str = enum_to_db_state(state);

        conn.execute(
            "UPDATE code_nodes SET current_state = ?1, updated_at = datetime('now') WHERE node_id = ?2",
            rusqlite::params![state_str, node_id],
        )
        .map_err(|e| e.to_string())?;

        // Also log to history
        conn.execute(
            r#"
            INSERT INTO code_state_history (node_id, state, valid_from, caused_by_event_type)
            VALUES (?1, ?2, datetime('now'), 'state_change')
            "#,
            rusqlite::params![node_id, state_str],
        )
        .map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Mark all tests as passed for nodes matching a test name pattern.
    ///
    /// Uses smart matching:
    /// 1. Test edges from graph
    /// 2. Name patterns: test_foo, should_foo, it_foo → foo
    /// 3. Compound names: test_foo_bar → foo_bar, foo, bar
    /// 4. Module paths: module::test_func → module::func
    pub fn mark_test_passed(&self, test_name: &str) -> Result<usize, String> {
        let conn = self.db.connection();

        // Try 1: Find nodes via test edges
        let affected = conn.execute(
            r#"
            UPDATE code_nodes
            SET current_state = 'tested_green',
                updated_at = datetime('now'),
                last_tested_at = datetime('now')
            WHERE node_id IN (
                SELECT to_node FROM code_edges
                WHERE kind = 'tests'
                AND from_node LIKE ?1
            )
            "#,
            rusqlite::params![format!("%{}%", test_name)],
        )
        .map_err(|e| e.to_string())?;

        if affected > 0 {
            return Ok(affected);
        }

        // Try 2: Smart name matching with multiple candidates
        let candidates = extract_test_candidates(test_name);
        let mut total_affected = 0;

        for candidate in candidates {
            let affected = conn.execute(
                r#"
                UPDATE code_nodes
                SET current_state = 'tested_green',
                    updated_at = datetime('now'),
                    last_tested_at = datetime('now')
                WHERE name = ?1
                AND kind IN ('rust_function', 'python_function', 'method', 'rust_struct', 'rust_enum', 'rust_impl', 'python_class', 'module')
                AND current_state != 'tested_green'
                "#,
                rusqlite::params![candidate],
            )
            .map_err(|e| e.to_string())?;

            total_affected += affected;
        }

        Ok(total_affected)
    }

    /// Mark test as failed for nodes matching a test name pattern.
    pub fn mark_test_failed(&self, test_name: &str) -> Result<usize, String> {
        let conn = self.db.connection();

        // Try 1: Find nodes via test edges
        let affected = conn.execute(
            r#"
            UPDATE code_nodes
            SET current_state = 'tested_red',
                updated_at = datetime('now'),
                last_tested_at = datetime('now')
            WHERE node_id IN (
                SELECT to_node FROM code_edges
                WHERE kind = 'tests'
                AND from_node LIKE ?1
            )
            "#,
            rusqlite::params![format!("%{}%", test_name)],
        )
        .map_err(|e| e.to_string())?;

        if affected > 0 {
            return Ok(affected);
        }

        // Try 2: Smart name matching with multiple candidates
        let candidates = extract_test_candidates(test_name);
        let mut total_affected = 0;

        for candidate in candidates {
            let affected = conn.execute(
                r#"
                UPDATE code_nodes
                SET current_state = 'tested_red',
                    updated_at = datetime('now'),
                    last_tested_at = datetime('now')
                WHERE name = ?1
                AND kind IN ('rust_function', 'python_function', 'method', 'rust_struct', 'rust_enum', 'rust_impl', 'python_class', 'module')
                "#,
                rusqlite::params![candidate],
            )
            .map_err(|e| e.to_string())?;

            total_affected += affected;
        }

        Ok(total_affected)
    }
}

/// Extract candidate function names from a test name.
///
/// Examples:
/// - "test_foo" → ["foo", "Foo"]
/// - "test_foo_bar" → ["foo_bar", "FooBar", "foo", "bar"]
/// - "test_state_tracker_new" → ["state_tracker_new", "StateTracker", "state_tracker", "new"]
fn extract_test_candidates(test_name: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    // Extract the function name part (after last ::)
    let fn_name = test_name.split("::").last().unwrap_or(test_name);

    // Strip common test prefixes
    let base = fn_name
        .strip_prefix("test_")
        .or_else(|| fn_name.strip_prefix("should_"))
        .or_else(|| fn_name.strip_prefix("it_"))
        .or_else(|| fn_name.strip_prefix("when_"))
        .or_else(|| fn_name.strip_prefix("given_"))
        .unwrap_or(fn_name);

    if base.is_empty() {
        return candidates;
    }

    // Add the full base name (snake_case)
    candidates.push(base.to_string());

    // Add PascalCase version
    candidates.push(to_pascal_case(base));

    // Split by underscore
    let parts: Vec<&str> = base.split('_').collect();
    if parts.len() > 1 {
        // Add each individual word
        for part in &parts {
            if !part.is_empty() && part.len() > 2 {
                candidates.push(part.to_string());
            }
        }

        // Add progressively longer prefixes in both cases
        // e.g., state_tracker_new → state, state_tracker, StateTracker
        for i in 0..parts.len() {
            let snake: String = parts[..=i].join("_");
            let pascal = to_pascal_case(&snake);
            if !candidates.contains(&snake) {
                candidates.push(snake);
            }
            if !candidates.contains(&pascal) {
                candidates.push(pascal);
            }
        }
    }

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|x| seen.insert(x.clone()));

    candidates
}

/// Convert snake_case to PascalCase
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

impl<'a> DbStateTracker<'a> {
    /// Get summary of all node states from database.
    pub fn summary(&self) -> StateSummary {
        let conn = self.db.connection();

        let mut stmt = conn
            .prepare("SELECT current_state, COUNT(*) FROM code_nodes GROUP BY current_state")
            .unwrap();

        let mut green = 0;
        let mut red = 0;
        let mut dirty = 0;
        let mut untested = 0;
        let mut total = 0;

        let rows = stmt
            .query_map([], |row| {
                let state: String = row.get(0)?;
                let count: i64 = row.get(1)?;
                Ok((state, count as usize))
            })
            .unwrap();

        for row in rows.flatten() {
            let (state, count) = row;
            total += count;
            match state.as_str() {
                "tested_green" => green += count,
                "tested_red" => red += count,
                "stale_direct" | "stale_transitive" | "stale_deep" | "changed" => dirty += count,
                "unknown" | "untouched" => untested += count,
                _ => untested += count,
            }
        }

        StateSummary {
            green,
            red,
            dirty,
            untested,
            total,
        }
    }

    /// Get all nodes in a specific state.
    pub fn nodes_in_state(&self, state: NodeState) -> Vec<String> {
        let conn = self.db.connection();
        let state_str = enum_to_db_state(state);

        let mut stmt = conn
            .prepare("SELECT node_id FROM code_nodes WHERE current_state = ?1")
            .unwrap();

        stmt.query_map([state_str], |row| row.get(0))
            .unwrap()
            .flatten()
            .collect()
    }

    /// Get nodes that need testing (dirty, untested, or red).
    pub fn nodes_needing_tests(&self) -> Vec<String> {
        let conn = self.db.connection();

        let mut stmt = conn
            .prepare(
                r#"
                SELECT node_id FROM code_nodes
                WHERE current_state IN ('unknown', 'untouched', 'changed',
                    'stale_direct', 'stale_transitive', 'stale_deep', 'tested_red')
                "#,
            )
            .unwrap();

        stmt.query_map([], |row| row.get(0))
            .unwrap()
            .flatten()
            .collect()
    }

    /// Initialize all unknown nodes to untested state.
    pub fn init_unknown_to_untested(&self) -> Result<usize, String> {
        let conn = self.db.connection();
        let affected = conn
            .execute(
                "UPDATE code_nodes SET current_state = 'untouched' WHERE current_state = 'unknown'",
                [],
            )
            .map_err(|e| e.to_string())?;
        Ok(affected)
    }
}

/// Convert database state string to NodeState enum.
fn db_state_to_enum(state: &str) -> NodeState {
    match state {
        "tested_green" | "qa_approved" => NodeState::Green,
        "tested_red" | "qa_flagged" => NodeState::Red,
        "stale_direct" | "stale_transitive" | "stale_deep" | "changed" => NodeState::Dirty,
        _ => NodeState::Untested,
    }
}

/// Convert NodeState enum to database state string.
fn enum_to_db_state(state: NodeState) -> &'static str {
    match state {
        NodeState::Green => "tested_green",
        NodeState::Red => "tested_red",
        NodeState::Dirty => "stale_direct",
        NodeState::Untested => "untouched",
    }
}
