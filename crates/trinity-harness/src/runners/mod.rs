//! Test runners for executing and parsing test results.
//!
//! Supports:
//! - Cargo test (Rust)
//! - Pytest (Python)
//! - Result mapping to code graph

mod cargo;
mod mapper;
mod pytest;

pub use cargo::{run_cargo_test, CargoTestConfig, CargoTestResult, TestOutcome, TestResult};
pub use mapper::{
    get_test_targets, lookup_test_node, map_results, MappingResult, NodeResult,
};
pub use pytest::{run_pytest, PytestConfig, PytestError, PytestResult};
