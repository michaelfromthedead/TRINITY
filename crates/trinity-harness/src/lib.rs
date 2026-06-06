//! Trinity Harness - Multi-language code analysis framework
//!
//! This crate provides infrastructure for parsing and analyzing code
//! across Rust, Python, and WGSL languages.

pub mod ci;
pub mod cli;
pub mod daemon;
pub mod db;
pub mod graph;
pub mod parsers;
pub mod runners;
pub mod state;

pub use ci::{
    generate_harness_steps, generate_yaml, validate_workflow, ValidationResult as CiValidationResult,
    WorkflowConfig, WorkflowStep,
};
pub use cli::{
    cmd_daemon, cmd_query_needs_testing, cmd_run_stale, cmd_update_from_results,
    execute_command, CliConfig, CommandResult, OutputFormat,
};
pub use daemon::{DaemonConfig, DaemonEvent, DaemonStatus, HarnessDaemon};
pub use db::HarnessDb;
pub use graph::{persist_graph_to_db, CodeEdge, CodeNode, GraphBuilder, PersistError, ScanError, ScanStats};
pub use parsers::{CodeUnit, ContentHashes, Language, ParserRegistry, UnitType};
pub use state::StateMachine;
