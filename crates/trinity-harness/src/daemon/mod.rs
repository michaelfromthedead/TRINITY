//! HarnessDaemon for continuous test monitoring.
//!
//! Provides a daemon that monitors file changes and maintains test state.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::graph::{CodeGraph, NodeId};
use crate::runners::{StateTracker, NodeState, TestEvent};

/// Configuration for the harness daemon.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Project root directory.
    pub project_root: String,
    /// How often to check for changes (in milliseconds).
    pub poll_interval_ms: u64,
    /// Debounce time for file changes (in milliseconds).
    pub debounce_ms: u64,
    /// Maximum events to process per tick.
    pub max_events_per_tick: usize,
    /// Enable verbose logging.
    pub verbose: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            project_root: ".".to_string(),
            poll_interval_ms: 1000,
            debounce_ms: 100,
            max_events_per_tick: 100,
            verbose: false,
        }
    }
}

impl DaemonConfig {
    /// Create a new config for a project.
    pub fn new(project_root: impl Into<String>) -> Self {
        Self {
            project_root: project_root.into(),
            ..Default::default()
        }
    }

    /// Set poll interval.
    pub fn poll_interval(mut self, ms: u64) -> Self {
        self.poll_interval_ms = ms;
        self
    }

    /// Set debounce time.
    pub fn debounce(mut self, ms: u64) -> Self {
        self.debounce_ms = ms;
        self
    }

    /// Enable verbose mode.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// Event from the daemon.
#[derive(Debug, Clone)]
pub enum DaemonEvent {
    /// File was created.
    FileCreated { path: String },
    /// File was modified.
    FileModified { path: String },
    /// File was deleted.
    FileDeleted { path: String },
    /// State changed for a node.
    StateChanged { node_id: NodeId, old: NodeState, new: NodeState },
    /// Daemon started.
    Started,
    /// Daemon stopped.
    Stopped,
    /// Error occurred.
    Error { message: String },
}

/// Callback for daemon events.
pub type EventCallback = Box<dyn Fn(DaemonEvent) + Send + Sync>;

/// The harness daemon.
pub struct HarnessDaemon {
    /// Configuration.
    config: DaemonConfig,
    /// State tracker.
    tracker: StateTracker,
    /// Running flag.
    running: Arc<AtomicBool>,
    /// Event callbacks.
    callbacks: Vec<EventCallback>,
    /// Last tick time.
    last_tick: Option<Instant>,
    /// Pending events.
    pending_events: Vec<DaemonEvent>,
}

impl HarnessDaemon {
    /// Create a new daemon.
    pub fn new(config: DaemonConfig) -> Self {
        Self {
            config,
            tracker: StateTracker::new(),
            running: Arc::new(AtomicBool::new(false)),
            callbacks: Vec::new(),
            last_tick: None,
            pending_events: Vec::new(),
        }
    }

    /// Add an event callback.
    pub fn on_event(&mut self, callback: EventCallback) {
        self.callbacks.push(callback);
    }

    /// Get the state tracker.
    pub fn tracker(&self) -> &StateTracker {
        &self.tracker
    }

    /// Get mutable state tracker.
    pub fn tracker_mut(&mut self) -> &mut StateTracker {
        &mut self.tracker
    }

    /// Check if the daemon is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get a stop handle.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }

    /// Start the daemon (blocking).
    pub fn run(&mut self) {
        self.running.store(true, Ordering::SeqCst);
        self.emit_event(DaemonEvent::Started);

        if self.config.verbose {
            eprintln!("[daemon] Started");
        }

        while self.running.load(Ordering::SeqCst) {
            self.tick();
            std::thread::sleep(Duration::from_millis(self.config.poll_interval_ms));
        }

        self.emit_event(DaemonEvent::Stopped);
        if self.config.verbose {
            eprintln!("[daemon] Stopped");
        }
    }

    /// Run a single tick.
    pub fn tick(&mut self) {
        let now = Instant::now();
        self.last_tick = Some(now);

        // Process pending events
        let events: Vec<_> = self.pending_events.drain(..).collect();
        for event in events.into_iter().take(self.config.max_events_per_tick) {
            self.process_event(event);
        }
    }

    /// Stop the daemon.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Queue a file event.
    pub fn queue_file_event(&mut self, event: DaemonEvent) {
        self.pending_events.push(event);
    }

    /// Process a single event.
    fn process_event(&mut self, event: DaemonEvent) {
        match &event {
            DaemonEvent::FileModified { path } => {
                if self.config.verbose {
                    eprintln!("[daemon] File modified: {}", path);
                }
                // In a real implementation, we'd look up the node and mark it dirty
            }
            DaemonEvent::FileCreated { path } => {
                if self.config.verbose {
                    eprintln!("[daemon] File created: {}", path);
                }
            }
            DaemonEvent::FileDeleted { path } => {
                if self.config.verbose {
                    eprintln!("[daemon] File deleted: {}", path);
                }
            }
            _ => {}
        }

        self.emit_event(event);
    }

    /// Handle a node state change.
    pub fn handle_state_change(&mut self, node_id: NodeId, new_state: NodeState) {
        let old_state = self.tracker.get_state(node_id);
        if old_state != new_state {
            self.tracker.set_state(node_id, new_state);
            self.emit_event(DaemonEvent::StateChanged {
                node_id,
                old: old_state,
                new: new_state,
            });
        }
    }

    /// Mark nodes as dirty (modified).
    pub fn mark_dirty(&mut self, node_ids: &[NodeId]) {
        for &node_id in node_ids {
            self.tracker.handle_event(TestEvent::CodeModified { node_id });
        }
    }

    /// Get nodes that need testing.
    pub fn needs_testing(&self, all_nodes: &[NodeId]) -> Vec<NodeId> {
        all_nodes
            .iter()
            .filter(|&&id| {
                let state = self.tracker.get_state(id);
                matches!(state, NodeState::Dirty | NodeState::Untested | NodeState::Red)
            })
            .copied()
            .collect()
    }

    /// Emit an event to all callbacks.
    fn emit_event(&self, event: DaemonEvent) {
        for callback in &self.callbacks {
            callback(event.clone());
        }
    }
}

/// Status of the daemon.
#[derive(Debug, Clone)]
pub struct DaemonStatus {
    /// Whether the daemon is running.
    pub running: bool,
    /// Total nodes tracked.
    pub total_nodes: usize,
    /// Nodes needing tests.
    pub needs_testing: usize,
    /// Last tick time (if available).
    pub last_tick_ms: Option<u64>,
}

impl DaemonStatus {
    /// Create a status from a daemon.
    pub fn from_daemon(daemon: &HarnessDaemon, node_ids: &[NodeId]) -> Self {
        Self {
            running: daemon.is_running(),
            total_nodes: node_ids.len(),
            needs_testing: daemon.needs_testing(node_ids).len(),
            last_tick_ms: daemon.last_tick.map(|t| t.elapsed().as_millis() as u64),
        }
    }
}
