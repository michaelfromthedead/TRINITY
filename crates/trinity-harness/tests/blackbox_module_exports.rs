//! Blackbox tests for T-HARNESS-1.1: Module structure verification
//!
//! These tests verify that the trinity-harness crate has the expected
//! module structure as defined in PHASE_1_INFRASTRUCTURE_ARCH.md:
//! - db module
//! - parsers module (rust/python/wgsl submodules)
//! - graph module (nodes/edges submodules)
//! - state module (machine submodule)
//!
//! Note: The ARCH specifies module structure exists internally. The public
//! contract requires types to be accessible, not necessarily the submodules
//! themselves being public. We test what the external user needs.

/// Test that the db module is publicly accessible
#[test]
fn db_module_exists() {
    // Attempt to use something from the db module
    // If the module doesn't exist or isn't public, this won't compile
    #[allow(unused_imports)]
    use trinity_harness::db;
}

/// Test that the parsers module is publicly accessible
#[test]
fn parsers_module_exists() {
    #[allow(unused_imports)]
    use trinity_harness::parsers;
}

/// Test that the graph module is publicly accessible
#[test]
fn graph_module_exists() {
    #[allow(unused_imports)]
    use trinity_harness::graph;
}

/// Test that the state module is publicly accessible
#[test]
fn state_module_exists() {
    #[allow(unused_imports)]
    use trinity_harness::state;
}
