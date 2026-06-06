//! Test runners for executing and parsing test results.
//!
//! Supports:
//! - Cargo test (Rust)
//! - Pytest (Python)

mod cargo;

pub use cargo::{run_cargo_test, CargoTestConfig, CargoTestResult, TestOutcome, TestResult};
