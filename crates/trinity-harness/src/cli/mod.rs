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
    use crate::runners::DbStateTracker;

    // Open database
    let db_path = ".harness/state.db";
    let db = match crate::db::HarnessDb::open(db_path) {
        Ok(db) => db,
        Err(e) => return CommandResult::err(format!("Failed to open database: {:?}", e)),
    };

    let tracker = DbStateTracker::new(&db);
    let needs_testing = tracker.nodes_needing_tests();
    let summary = tracker.summary();

    let mut output = String::new();

    match config.format {
        OutputFormat::Text => {
            output.push_str(&format!(
                "Nodes needing testing: {}/{}\n\n",
                needs_testing.len(),
                summary.total
            ));

            output.push_str(&format!("State summary:\n"));
            output.push_str(&format!("  GREEN:    {}\n", summary.green));
            output.push_str(&format!("  RED:      {}\n", summary.red));
            output.push_str(&format!("  DIRTY:    {}\n", summary.dirty));
            output.push_str(&format!("  UNTESTED: {}\n", summary.untested));

            if !needs_testing.is_empty() {
                output.push_str(&format!("\nFirst 20 nodes needing tests:\n"));
                for node_id in needs_testing.iter().take(20) {
                    output.push_str(&format!("  {}\n", node_id));
                }
                if needs_testing.len() > 20 {
                    output.push_str(&format!("  ... and {} more\n", needs_testing.len() - 20));
                }
            }
        }
        OutputFormat::Json => {
            output = serde_json::to_string_pretty(&serde_json::json!({
                "needs_testing": needs_testing.len(),
                "total": summary.total,
                "summary": {
                    "green": summary.green,
                    "red": summary.red,
                    "dirty": summary.dirty,
                    "untested": summary.untested
                },
                "nodes": needs_testing
            }))
            .unwrap_or_else(|_| "{}".to_string());
        }
    }

    CommandResult::ok(output)
}

/// Run only stale tests.
pub fn cmd_run_stale(config: &CliConfig, package: Option<&str>, test_filter: Option<&str>) -> CommandResult {
    use crate::runners::DbStateTracker;

    // Open database
    let db_path = ".harness/state.db";
    let db = match crate::db::HarnessDb::open(db_path) {
        Ok(db) => db,
        Err(e) => return CommandResult::err(format!("Failed to open database: {:?}", e)),
    };

    let tracker = DbStateTracker::new(&db);
    let needs_testing = tracker.nodes_needing_tests();

    if needs_testing.is_empty() {
        return CommandResult::ok("No stale tests to run");
    }

    let project_root = config.project_root.to_string_lossy().to_string();
    let mut exec_config = ExecutorConfig::new(&project_root);

    // If package specified, only test that package
    if let Some(pkg) = package {
        exec_config = exec_config.package(pkg);
    }

    // If test filter specified, only run matching tests
    if let Some(filter) = test_filter {
        exec_config = exec_config.test_filter(filter);
    }

    let result = run_all_tests(&exec_config);

    // Update state in database based on test results
    let mut passed_count = 0;
    let mut failed_count = 0;

    if let Some(ref cargo) = result.cargo {
        for test in &cargo.tests {
            match test.outcome {
                crate::runners::TestOutcome::Passed => {
                    if tracker.mark_test_passed(&test.name).is_ok() {
                        passed_count += 1;
                    }
                }
                crate::runners::TestOutcome::Failed => {
                    if tracker.mark_test_failed(&test.name).is_ok() {
                        failed_count += 1;
                    }
                }
                _ => {}
            }
        }
    }

    let summary = tracker.summary();
    let msg = format!(
        "Tests completed:\n  Cargo: {} passed, {} failed\n  Pytest: {} passed, {} failed\n\nState updated:\n  GREEN: {}\n  RED: {}\n  DIRTY: {}\n  UNTESTED: {}",
        result.cargo.as_ref().map(|r| r.passed).unwrap_or(0),
        result.cargo.as_ref().map(|r| r.failed).unwrap_or(0),
        result.pytest.as_ref().map(|r| r.passed).unwrap_or(0),
        result.pytest.as_ref().map(|r| r.failed).unwrap_or(0),
        summary.green,
        summary.red,
        summary.dirty,
        summary.untested,
    );
    CommandResult::ok(msg)
}

/// Update state from test results.
pub fn cmd_update_from_results(
    config: &CliConfig,
    results_path: Option<&str>,
) -> CommandResult {
    use crate::runners::DbStateTracker;

    // Open database
    let db_path = ".harness/state.db";
    let db = match crate::db::HarnessDb::open(db_path) {
        Ok(db) => db,
        Err(e) => return CommandResult::err(format!("Failed to open database: {:?}", e)),
    };

    let tracker = DbStateTracker::new(&db);

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
        // Run tests and update database
        let project_root = config.project_root.to_string_lossy().to_string();
        let exec_config = ExecutorConfig::new(&project_root);

        let result = run_all_tests(&exec_config);

        // Update state in database based on test results
        if let Some(ref cargo) = result.cargo {
            for test in &cargo.tests {
                match test.outcome {
                    crate::runners::TestOutcome::Passed => {
                        let _ = tracker.mark_test_passed(&test.name);
                    }
                    crate::runners::TestOutcome::Failed => {
                        let _ = tracker.mark_test_failed(&test.name);
                    }
                    _ => {}
                }
            }
        }

        let summary = tracker.summary();
        let msg = format!(
            "State updated:\n  GREEN: {}\n  RED: {}\n  DIRTY: {}\n  UNTESTED: {}",
            summary.green, summary.red, summary.dirty, summary.untested
        );
        CommandResult::ok(msg)
    }
}

/// Scan source directories and build the code graph.
pub fn cmd_scan(paths: &[String]) -> CommandResult {
    use std::path::Path;
    use crate::graph::{ScanStats, create_test_edges, CombinedMapper};
    use crate::parsers::Language;

    if paths.is_empty() {
        return CommandResult::err("Usage: scan <path> [path...]");
    }

    let registry = ParserRegistry::new();
    let builder = GraphBuilder::new(&registry);
    let mut total_stats = ScanStats::default();

    // Phase 1: Scan all paths into a single graph
    let mut graph = crate::graph::CodeGraph::new();

    for path in paths {
        match builder.full_scan(Path::new(path)) {
            Ok((partial_graph, stats)) => {
                total_stats.files_scanned += stats.files_scanned;
                total_stats.files_skipped += stats.files_skipped;
                total_stats.total_nodes += stats.total_nodes;
                for (lang, count) in stats.nodes_per_language {
                    *total_stats.nodes_per_language.entry(lang).or_insert(0) += count;
                }
                // Merge nodes into main graph
                for node in partial_graph.nodes() {
                    graph.add_node(node.clone());
                }
            }
            Err(e) => {
                return CommandResult::err(format!("Scan error for {}: {:?}", path, e));
            }
        }
    }

    // Phase 2: Analyze dependencies and create edges
    let mut dep_edges = 0;
    let mut test_edges = 0;

    for path in paths {
        // Analyze code dependencies (imports, calls)
        match builder.analyze_dependencies(Path::new(path), &mut graph) {
            Ok(stats) => {
                dep_edges += stats.deps_resolved;
            }
            Err(e) => {
                eprintln!("Warning: dependency analysis failed for {}: {:?}", path, e);
            }
        }
    }

    // Phase 3: Create test mappings
    let mapper = CombinedMapper::convention_only();
    // Use first path as root for test mapping
    let root = Path::new(&paths[0]);
    let (mappings, _mapping_stats) = mapper.map_tests(&graph, root);
    test_edges = create_test_edges(&mut graph, &mappings);

    // Phase 4: Persist to database
    let db_path = ".harness/state.db";
    std::fs::create_dir_all(".harness").ok();

    let db = match crate::db::HarnessDb::open(db_path) {
        Ok(db) => db,
        Err(e) => {
            return CommandResult::err(format!("Failed to open database: {:?}", e));
        }
    };

    let persist_stats = match crate::graph::persist_full_graph(&graph, &db) {
        Ok(stats) => stats,
        Err(e) => {
            return CommandResult::err(format!("Failed to persist graph: {:?}", e));
        }
    };

    let rust_count = total_stats.nodes_per_language.get(&Language::Rust).unwrap_or(&0);
    let python_count = total_stats.nodes_per_language.get(&Language::Python).unwrap_or(&0);
    let wgsl_count = total_stats.nodes_per_language.get(&Language::Wgsl).unwrap_or(&0);

    let msg = format!(
        "Scanned {} files, created {} nodes, {} edges ({} deps, {} tests)\n  Rust: {}\n  Python: {}\n  WGSL: {}\n  Skipped: {}\nPersisted to {}",
        total_stats.files_scanned,
        persist_stats.nodes,
        persist_stats.edges,
        dep_edges,
        test_edges,
        rust_count,
        python_count,
        wgsl_count,
        total_stats.files_skipped,
        db_path
    );
    CommandResult::ok(msg)
}

/// Show current state summary from database.
pub fn cmd_status() -> CommandResult {
    use crate::runners::DbStateTracker;

    let db_path = ".harness/state.db";

    if !std::path::Path::new(db_path).exists() {
        return CommandResult::err("No state.db found. Run 'scan' first.");
    }

    let db = match crate::db::HarnessDb::open(db_path) {
        Ok(db) => db,
        Err(e) => return CommandResult::err(format!("Failed to open database: {:?}", e)),
    };

    let tracker = DbStateTracker::new(&db);
    let summary = tracker.summary();

    let health = if summary.total > 0 {
        (summary.green as f64 / summary.total as f64) * 100.0
    } else {
        0.0
    };

    let msg = format!(
        "=== Trinity Harness Status ===\n\nTotal nodes: {}\n\n  GREEN:    {:>5} nodes\n  RED:      {:>5} nodes\n  DIRTY:    {:>5} nodes\n  UNTESTED: {:>5} nodes\n\nHealth: {:.1}%",
        summary.total,
        summary.green,
        summary.red,
        summary.dirty,
        summary.untested,
        health
    );
    CommandResult::ok(msg)
}

/// Smart test execution options.
#[derive(Debug, Clone, Default)]
pub struct SmartTestOptions {
    pub package: Option<String>,
    pub incremental: bool,
    pub affected_only: bool,
    pub priority: bool,
    pub disk_budget_gb: Option<f64>,
    pub use_cache: bool,
}

/// Run tests with smart execution strategy.
pub fn cmd_smart_test(config: &CliConfig, opts: SmartTestOptions) -> CommandResult {
    use crate::runners::{run_smart_tests, get_changed_files, SmartTestConfig, DbStateTracker};

    let db_path = ".harness/state.db";
    let db = match crate::db::HarnessDb::open(db_path) {
        Ok(db) => db,
        Err(e) => return CommandResult::err(format!("Failed to open database: {:?}", e)),
    };

    let project_root = config.project_root.to_string_lossy().to_string();
    let mut smart_config = SmartTestConfig::new(&project_root);

    if let Some(pkg) = opts.package {
        smart_config = smart_config.package(pkg);
    }

    if opts.incremental {
        smart_config = smart_config.incremental();
    }

    if opts.affected_only {
        let changed = get_changed_files(&project_root);
        smart_config = smart_config.affected_only(changed);
    }

    if opts.priority {
        smart_config = smart_config.priority_order();
    }

    if let Some(budget_gb) = opts.disk_budget_gb {
        let bytes = (budget_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        smart_config = smart_config.disk_budget(bytes);
    }

    if opts.use_cache {
        smart_config = smart_config.use_cache();
    }

    let result = run_smart_tests(&smart_config, &db);

    // Update state based on results
    let tracker = DbStateTracker::new(&db);
    for test in &result.tests {
        match test.outcome {
            crate::runners::TestOutcome::Passed => {
                let _ = tracker.mark_test_passed(&test.name);
            }
            crate::runners::TestOutcome::Failed => {
                let _ = tracker.mark_test_failed(&test.name);
            }
            _ => {}
        }
    }

    let summary = tracker.summary();
    let disk_mb = result.disk_reclaimed as f64 / (1024.0 * 1024.0);

    let msg = format!(
        "Smart Test Results:\n  Files tested: {}\n  Tests run: {}\n  Passed: {}\n  Failed: {}\n  Skipped (cached): {}\n  Skipped (unaffected): {}\n  Disk reclaimed: {:.1} MB\n\nState:\n  GREEN: {}\n  RED: {}\n  DIRTY: {}\n  UNTESTED: {}",
        result.files_processed.len(),
        result.total_run,
        result.passed,
        result.failed,
        result.skipped_cached,
        result.skipped_unaffected,
        disk_mb,
        summary.green,
        summary.red,
        summary.dirty,
        summary.untested,
    );

    if !result.errors.is_empty() {
        let errors: String = result.errors.iter().map(|e| format!("\n  - {}", e)).collect();
        return CommandResult::ok(format!("{}\n\nErrors:{}", msg, errors));
    }

    CommandResult::ok(msg)
}

/// Parse and execute a CLI command.
pub fn execute_command(args: &[String]) -> CommandResult {
    if args.is_empty() {
        return CommandResult::err("Usage: trinity-harness <command> [args]\n\nCommands:\n  scan <paths...>       Scan source directories\n  status                Show state summary\n  run-stale [-p pkg]    Run stale tests\n  smart [flags]         Smart test execution\n    --incremental       Run one file at a time, clean between\n    --affected          Only run tests for changed files\n    --priority          Run recently-changed tests first\n    --budget <GB>       Stay within disk budget\n    --cache             Skip tests for unchanged code\n  update [file]         Update from test results\n  daemon                Start file watcher");
    }

    let config = CliConfig::default();

    match args[0].as_str() {
        "scan" => cmd_scan(&args[1..].to_vec()),
        "status" => cmd_status(),
        "daemon" => cmd_daemon(&config),
        "query" => {
            if args.len() > 1 && args[1] == "needs-testing" {
                cmd_query_needs_testing(&config)
            } else {
                CommandResult::err("Unknown query. Use: query needs-testing")
            }
        }
        "run-stale" => {
            // Parse arguments: run-stale [-p package] [-t test_filter]
            let mut package = None;
            let mut test_filter = None;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "-p" if i + 1 < args.len() => {
                        package = Some(args[i + 1].as_str());
                        i += 2;
                    }
                    "-t" if i + 1 < args.len() => {
                        test_filter = Some(args[i + 1].as_str());
                        i += 2;
                    }
                    _ => i += 1,
                }
            }
            cmd_run_stale(&config, package, test_filter)
        }
        "update" | "update-from-results" => {
            let path = args.get(1).map(|s| s.as_str());
            cmd_update_from_results(&config, path)
        }
        "smart" => {
            // Parse: smart [-p pkg] [--incremental] [--affected] [--priority] [--budget GB] [--cache]
            let mut opts = SmartTestOptions::default();
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "-p" | "--package" if i + 1 < args.len() => {
                        opts.package = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--incremental" => {
                        opts.incremental = true;
                        i += 1;
                    }
                    "--affected" => {
                        opts.affected_only = true;
                        i += 1;
                    }
                    "--priority" => {
                        opts.priority = true;
                        i += 1;
                    }
                    "--budget" if i + 1 < args.len() => {
                        opts.disk_budget_gb = args[i + 1].parse().ok();
                        i += 2;
                    }
                    "--cache" => {
                        opts.use_cache = true;
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
            cmd_smart_test(&config, opts)
        }
        _ => CommandResult::err(format!("Unknown command: {}. Use: scan, status, smart, run-stale, update, daemon", args[0])),
    }
}
