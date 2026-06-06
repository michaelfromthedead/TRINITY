//! Constants for trinity-harness configuration defaults.

/// Default timeout for cargo test in seconds (10 minutes).
pub const DEFAULT_CARGO_TIMEOUT_SECS: u64 = 600;

/// Default timeout for pytest in seconds (30 minutes).
pub const DEFAULT_PYTEST_TIMEOUT_SECS: u64 = 1800;

/// Default poll interval for daemon in milliseconds.
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 1000;

/// Default debounce time for file watcher in milliseconds.
pub const DEFAULT_DEBOUNCE_MS: u64 = 100;

/// Default poll interval for file watcher in milliseconds.
pub const DEFAULT_WATCHER_POLL_INTERVAL_MS: u64 = 500;

/// Maximum events to process per daemon tick.
pub const MAX_EVENTS_PER_TICK: usize = 100;

/// Maximum depth for staleness propagation.
pub const MAX_PROPAGATION_DEPTH: usize = 10;

/// Maximum transition log size before rotation.
pub const MAX_LOG_SIZE: usize = 1000;

/// Maximum notification buffer size.
pub const MAX_BUFFER_SIZE: usize = 10000;

/// SQLite cache size (negative = KB).
pub const SQLITE_CACHE_SIZE: i32 = -64000;
