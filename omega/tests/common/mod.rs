//! Shared test utilities and mock RHI backend.
//!
//! This module provides a mock/headless RHI implementation that mirrors the
//! Python `Null*` classes from `engine/platform/rhi/`. Using this mock backend,
//! all RHI integration tests can run without any GPU hardware.

mod mock;

// Re-export everything from mock for convenience
pub use mock::*;
