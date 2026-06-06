//! Test runners for executing and parsing test results.
//!
//! Supports:
//! - Cargo test (Rust)
//! - Pytest (Python)
//! - Result mapping to code graph
//! - State transitions based on test results
//! - Unified test execution
//! - Baseline recording
//! - Validation and reporting

mod baseline;
mod cargo;
mod executor;
mod mapper;
mod pytest;
mod smart;
mod transitions;
mod validation;

pub use baseline::{
    compare_baselines, record_baseline, Baseline, BaselineComparison, BaselineSummary,
    NodeStateRecord, StateChange, TestFailure,
};
pub use cargo::{run_cargo_test, CargoTestConfig, CargoTestResult, TestOutcome, TestResult};
pub use executor::{
    estimate_duration, run_all_tests, should_skip_tests, ExecutorConfig, ExecutorResult,
};
pub use mapper::{
    get_test_targets, lookup_test_node, map_results, MappingResult, NodeResult,
};
pub use pytest::{run_pytest, PytestConfig, PytestError, PytestResult};
pub use transitions::{
    DbStateTracker, NodeState, StateTracker, StateSummary, StateTransition, TestEvent,
};
pub use smart::{
    get_available_disk_space, get_changed_files, run_smart_tests, SmartTestConfig,
    SmartTestResult,
};
pub use validation::{
    generate_summary, validate_and_summarize, validate_baseline, validate_tracker,
    ValidationResult,
};
