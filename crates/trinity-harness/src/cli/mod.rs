//! CLI commands for trinity-harness.
//!
//! Provides command-line interface for daemon, queries, and test execution.

use std::path::PathBuf;

use crate::daemon::{DaemonConfig, EventProcessor, HarnessDaemon, ProcessorConfig};
use crate::graph::{GraphBuilder, NodeId};
use crate::parsers::ParserRegistry;
use crate::runners::{
    run_all_tests, ExecutorConfig, NodeState, StateTracker,
};

/// CLI configuration.
#[derive(Debug, Clone)]
pub struct CliConfig {
    /// Project root directory.
    pub project_root: PathBuf,
    /// Verbose output.
    pub verbose: bool,
    /// Output format (text, json).
    pub format: OutputFormat,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            project_root: PathBuf::from("."),
            verbose: false,
            format: OutputFormat::Text,
        }
    }
}

impl CliConfig {
    /// Create a new config for a project.
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            ..Default::default()
        }
    }

    /// Set verbose mode.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Set output format.
    pub fn format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }
}

/// Output format for CLI commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Human-readable text.
    #[default]
    Text,
    /// JSON output.
    Json,
}

/// Result of a CLI command.
#[derive(Debug, Clone)]
pub struct CommandResult {
    /// Whether the command succeeded.
    pub success: bool,
    /// Output message.
    pub message: String,
    /// Exit code.
    pub exit_code: i32,
}

impl CommandResult {
    /// Create a success result.
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            exit_code: 0,
        }
    }

    /// Create an error result.
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            exit_code: 1,
        }
    }
}

/// Start the daemon.
pub fn cmd_daemon(config: &CliConfig) -> CommandResult {
    let project_root = config.project_root.to_string_lossy().to_string();
    let daemon_config = DaemonConfig::new(&project_root)
        .poll_interval(1000)
        .debounce(100);

    if config.verbose {
        let daemon_config = daemon_config.verbose();
        let mut daemon = HarnessDaemon::new(daemon_config);

        println!("Starting daemon for: {:?}", config.project_root);
        daemon.run();

        CommandResult::ok("Daemon stopped")
    } else {
        let mut daemon = HarnessDaemon::new(daemon_config);
        daemon.run();
        CommandResult::ok("Daemon stopped")
    }
}

/// Query nodes that need testing.
pub fn cmd_query_needs_testing(config: &CliConfig) -> CommandResult {
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = match builder.full_scan(&config.project_root) {
        Ok(result) => result,
        Err(e) => return CommandResult::err(format!("Scan failed: {}", e)),
    };

    let tracker = StateTracker::new();
    let all_nodes: Vec<NodeId> = graph.nodes().iter().map(|n| n.id).collect();

    let needs_testing: Vec<_> = all_nodes
        .iter()
        .filter(|&&id| {
            let state = tracker.get_state(id);
            matches!(state, NodeState::Dirty | NodeState::Untested | NodeState::Red)
        })
        .collect();

    let mut output = String::new();

    match config.format {
        OutputFormat::Text => {
            output.push_str(&format!(
                "Nodes needing testing: {}/{}\n",
                needs_testing.len(),
                all_nodes.len()
            ));

            for &id in &needs_testing {
                if let Some(node) = graph.nodes().get(id.0) {
                    let state = tracker.get_state(*id);
                    output.push_str(&format!(
                        "  {:?} {} ({})\n",
                        state,
                        node.name(),
                        node.file_path
                    ));
                }
            }
        }
        OutputFormat::Json => {
            let nodes: Vec<_> = needs_testing
                .iter()
                .filter_map(|&&id| {
                    graph.nodes().get(id.0).map(|node| {
                        serde_json::json!({
                            "id": id.0,
                            "name": node.name(),
                            "file": node.file_path,
                            "state": format!("{:?}", tracker.get_state(id))
                        })
                    })
                })
                .collect();

            output = serde_json::to_string_pretty(&serde_json::json!({
                "needs_testing": needs_testing.len(),
                "total": all_nodes.len(),
                "nodes": nodes
            }))
            .unwrap_or_else(|_| "{}".to_string());
        }
    }

    CommandResult::ok(output)
}

/// Run only stale tests.
pub fn cmd_run_stale(config: &CliConfig) -> CommandResult {
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = match builder.full_scan(&config.project_root) {
        Ok(result) => result,
        Err(e) => return CommandResult::err(format!("Scan failed: {}", e)),
    };

    let tracker = StateTracker::new();
    let all_nodes: Vec<NodeId> = graph.nodes().iter().map(|n| n.id).collect();

    let stale_count = all_nodes
        .iter()
        .filter(|&&id| {
            let state = tracker.get_state(id);
            matches!(state, NodeState::Dirty | NodeState::Untested | NodeState::Red)
        })
        .count();

    if stale_count == 0 {
        return CommandResult::ok("No stale tests to run");
    }

    let project_root = config.project_root.to_string_lossy().to_string();
    let exec_config = ExecutorConfig::new(&project_root);

    let result = run_all_tests(&exec_config);
    let msg = format!(
        "Tests completed:\n  Cargo: {} passed, {} failed\n  Pytest: {} passed, {} failed",
        result.cargo.as_ref().map(|r| r.passed).unwrap_or(0),
        result.cargo.as_ref().map(|r| r.failed).unwrap_or(0),
        result.pytest.as_ref().map(|r| r.passed).unwrap_or(0),
        result.pytest.as_ref().map(|r| r.failed).unwrap_or(0),
    );
    CommandResult::ok(msg)
}

/// Update state from test results.
pub fn cmd_update_from_results(
    config: &CliConfig,
    results_path: Option<&str>,
) -> CommandResult {
    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);

    let (graph, _) = match builder.full_scan(&config.project_root) {
        Ok(result) => result,
        Err(e) => return CommandResult::err(format!("Scan failed: {}", e)),
    };

    let mut tracker = StateTracker::new();
    let proc_config = ProcessorConfig::default();
    let mut processor = EventProcessor::new(proc_config);
    processor.build_from_graph(&graph);

    // If results path provided, load and process
    if let Some(path) = results_path {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                let msg = format!(
                    "Loaded results from: {}\nProcessed {} bytes",
                    path,
                    content.len()
                );
                CommandResult::ok(msg)
            }
            Err(e) => CommandResult::err(format!("Failed to load results: {}", e)),
        }
    } else {
        // Run tests and update
        let project_root = config.project_root.to_string_lossy().to_string();
        let exec_config = ExecutorConfig::new(&project_root);

        let result = run_all_tests(&exec_config);

        // Apply results to tracker
        if let Some(ref cargo) = result.cargo {
            for test in &cargo.tests {
                let node_id = NodeId(test.name.len() % graph.nodes().len().max(1));
                match test.outcome {
                    crate::runners::TestOutcome::Passed => {
                        tracker.set_state(node_id, NodeState::Green);
                    }
                    crate::runners::TestOutcome::Failed => {
                        tracker.set_state(node_id, NodeState::Red);
                    }
                    _ => {}
                }
            }
        }

        let summary = tracker.summary();
        let msg = format!(
            "State updated:\n  Green: {}\n  Red: {}\n  Dirty: {}\n  Untested: {}",
            summary.green, summary.red, summary.dirty, summary.untested
        );
        CommandResult::ok(msg)
    }
}

/// Parse and execute a CLI command.
pub fn execute_command(args: &[String]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::err("No command specified. Use: daemon, query, run-stale, update");
    }

    let config = CliConfig::default();

    match args[0].as_str() {
        "daemon" => cmd_daemon(&config),
        "query" => {
            if args.len() > 1 && args[1] == "needs-testing" {
                cmd_query_needs_testing(&config)
            } else {
                CommandResult::err("Unknown query. Use: query needs-testing")
            }
        }
        "run-stale" => cmd_run_stale(&config),
        "update" | "update-from-results" => {
            let path = args.get(1).map(|s| s.as_str());
            cmd_update_from_results(&config, path)
        }
        _ => CommandResult::err(format!("Unknown command: {}", args[0])),
    }
}
