//! Trinity Harness - Multi-language code analysis framework
//!
//! This crate provides infrastructure for parsing and analyzing code
//! across Rust, Python, and WGSL languages.

pub mod db;
pub mod graph;
pub mod parsers;
pub mod state;

pub use db::HarnessDb;
pub use graph::{CodeEdge, CodeNode, GraphBuilder, ScanError, ScanStats};
pub use parsers::{CodeUnit, ContentHashes, Language, ParserRegistry, UnitType};
pub use state::StateMachine;
