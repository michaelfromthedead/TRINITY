//! Event processor for handling file changes and state transitions.
//!
//! Processes file events and triggers appropriate state changes.

use std::collections::{HashMap, HashSet};

use crate::graph::{CodeGraph, NodeId};
use crate::runners::{NodeState, StateTracker, TestEvent};

use super::{ChangeKind, DaemonEvent, FileChange};

/// Configuration for the event processor.
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// Whether to propagate staleness to dependents.
    pub propagate_staleness: bool,
    /// Maximum depth for staleness propagation.
    pub max_propagation_depth: usize,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            propagate_staleness: true,
            max_propagation_depth: crate::constants::MAX_PROPAGATION_DEPTH,
        }
    }
}

/// Event processor for handling file changes.
pub struct EventProcessor {
    /// Configuration.
    config: ProcessorConfig,
    /// File path to node ID mapping.
    file_to_nodes: HashMap<String, Vec<NodeId>>,
    /// Node dependencies (node -> nodes that depend on it).
    dependents: HashMap<NodeId, Vec<NodeId>>,
    /// Processed events count.
    events_processed: usize,
}

impl EventProcessor {
    /// Create a new event processor.
    pub fn new(config: ProcessorConfig) -> Self {
        Self {
            config,
            file_to_nodes: HashMap::new(),
            dependents: HashMap::new(),
            events_processed: 0,
        }
    }

    /// Register a file to node mapping.
    pub fn register_file(&mut self, file_path: &str, node_id: NodeId) {
        self.file_to_nodes
            .entry(file_path.to_string())
            .or_default()
            .push(node_id);
    }

    /// Register a dependency (dependent depends on source).
    pub fn register_dependency(&mut self, source: NodeId, dependent: NodeId) {
        self.dependents
            .entry(source)
            .or_default()
            .push(dependent);
    }

    /// Build mappings from a code graph.
    pub fn build_from_graph(&mut self, graph: &CodeGraph) {
        self.file_to_nodes.clear();
        self.dependents.clear();

        // Map files to nodes
        for node in graph.nodes() {
            self.register_file(&node.file_path, node.id);
        }

        // Map dependencies
        for edge in graph.edges() {
            self.register_dependency(edge.target, edge.source);
        }
    }

    /// Process a file change event.
    pub fn process_file_change(
        &mut self,
        change: &FileChange,
        tracker: &mut StateTracker,
    ) -> ProcessResult {
        self.events_processed += 1;

        let path_str = change.path.to_string_lossy().to_string();
        let mut result = ProcessResult::new();

        // Find affected nodes
        let affected_nodes = self.find_affected_nodes(&path_str);
        result.directly_affected = affected_nodes.len();

        // Process each affected node
        for &node_id in &affected_nodes {
            match change.kind {
                ChangeKind::Modified | ChangeKind::Created => {
                    tracker.handle_event(TestEvent::CodeModified { node_id });
                    result.nodes_marked_dirty.push(node_id);
                }
                ChangeKind::Deleted => {
                    tracker.handle_event(TestEvent::Reset { node_id });
                    result.nodes_reset.push(node_id);
                }
            }
        }

        // Propagate staleness
        if self.config.propagate_staleness {
            let propagated = self.propagate_staleness(&affected_nodes, tracker);
            result.indirectly_affected = propagated.len();
            result.nodes_marked_dirty.extend(propagated);
        }

        result
    }

    /// Process a daemon event.
    pub fn process_event(
        &mut self,
        event: &DaemonEvent,
        tracker: &mut StateTracker,
    ) -> Option<ProcessResult> {
        match event {
            DaemonEvent::FileCreated { path } => {
                let change = FileChange {
                    path: path.into(),
                    kind: ChangeKind::Created,
                    timestamp: std::time::Instant::now(),
                };
                Some(self.process_file_change(&change, tracker))
            }
            DaemonEvent::FileModified { path } => {
                let change = FileChange {
                    path: path.into(),
                    kind: ChangeKind::Modified,
                    timestamp: std::time::Instant::now(),
                };
                Some(self.process_file_change(&change, tracker))
            }
            DaemonEvent::FileDeleted { path } => {
                let change = FileChange {
                    path: path.into(),
                    kind: ChangeKind::Deleted,
                    timestamp: std::time::Instant::now(),
                };
                Some(self.process_file_change(&change, tracker))
            }
            _ => None,
        }
    }

    /// Find nodes affected by a file change.
    fn find_affected_nodes(&self, file_path: &str) -> Vec<NodeId> {
        // Try exact match first
        if let Some(nodes) = self.file_to_nodes.get(file_path) {
            return nodes.clone();
        }

        // Try normalized path
        let normalized = normalize_path(file_path);
        if let Some(nodes) = self.file_to_nodes.get(&normalized) {
            return nodes.clone();
        }

        // Try suffix match
        for (registered, nodes) in &self.file_to_nodes {
            if registered.ends_with(file_path) || file_path.ends_with(registered) {
                return nodes.clone();
            }
        }

        Vec::new()
    }

    /// Propagate staleness to dependent nodes.
    fn propagate_staleness(
        &self,
        source_nodes: &[NodeId],
        tracker: &mut StateTracker,
    ) -> Vec<NodeId> {
        let mut propagated = Vec::new();
        let mut visited = HashSet::new();
        let mut to_visit: Vec<(NodeId, usize)> = source_nodes
            .iter()
            .map(|&id| (id, 0))
            .collect();

        while let Some((node_id, depth)) = to_visit.pop() {
            if depth >= self.config.max_propagation_depth {
                continue;
            }

            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id);

            // Get dependents
            if let Some(deps) = self.dependents.get(&node_id) {
                for &dep_id in deps {
                    let current_state = tracker.get_state(dep_id);
                    if current_state == NodeState::Green {
                        tracker.handle_event(TestEvent::CodeModified { node_id: dep_id });
                        propagated.push(dep_id);
                        to_visit.push((dep_id, depth + 1));
                    }
                }
            }
        }

        propagated
    }

    /// Get the number of events processed.
    pub fn events_processed(&self) -> usize {
        self.events_processed
    }

    /// Get the number of registered files.
    pub fn registered_files(&self) -> usize {
        self.file_to_nodes.len()
    }

    /// Get the number of registered dependencies.
    pub fn registered_dependencies(&self) -> usize {
        self.dependents.values().map(|v| v.len()).sum()
    }
}

/// Result of processing an event.
#[derive(Debug, Clone, Default)]
pub struct ProcessResult {
    /// Number of directly affected nodes.
    pub directly_affected: usize,
    /// Number of indirectly affected nodes (via propagation).
    pub indirectly_affected: usize,
    /// Nodes marked as dirty.
    pub nodes_marked_dirty: Vec<NodeId>,
    /// Nodes that were reset.
    pub nodes_reset: Vec<NodeId>,
}

impl ProcessResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total affected nodes.
    pub fn total_affected(&self) -> usize {
        self.directly_affected + self.indirectly_affected
    }
}

/// Normalize a file path.
fn normalize_path(path: &str) -> String {
    path.replace("\\", "/")
        .trim_start_matches("./")
        .to_string()
}

/// Process multiple events in batch.
pub fn process_batch(
    processor: &mut EventProcessor,
    events: &[DaemonEvent],
    tracker: &mut StateTracker,
) -> BatchResult {
    let mut result = BatchResult::new();

    for event in events {
        if let Some(r) = processor.process_event(event, tracker) {
            result.events_processed += 1;
            result.total_affected += r.total_affected();
        }
    }

    result
}

/// Result of batch processing.
#[derive(Debug, Clone, Default)]
pub struct BatchResult {
    /// Number of events processed.
    pub events_processed: usize,
    /// Total affected nodes.
    pub total_affected: usize,
}

impl BatchResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }
}
