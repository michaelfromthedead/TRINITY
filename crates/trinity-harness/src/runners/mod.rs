//! Test runners for executing and parsing test results.
//!
//! Supports:
//! - Cargo test (Rust)
//! - Pytest (Python)

mod cargo;
mod pytest;

pub use cargo::{run_cargo_test, CargoTestConfig, CargoTestResult, TestOutcome, TestResult};
pub use pytest::{run_pytest, PytestConfig, PytestError, PytestResult};
