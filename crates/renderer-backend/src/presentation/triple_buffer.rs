//! Triple buffering support for the TRINITY presentation system.
//!
//! This module provides flexible buffer management strategies for frame presentation,
//! supporting double, triple, quadruple, and custom buffer counts.
//!
//! # Architecture
//!
//! ```text
//! TripleBufferManager
//!     |-- BufferStrategy (Double, Triple, Quadruple, Custom)
//!     |-- BufferSlot[] (Available, Rendering, Presenting, Queued)
//!     |-- BufferStats (usage metrics and diagnostics)
//!     `-- State machine for buffer lifecycle
//! ```
//!
//! # Buffer Lifecycle
//!
//! ```text
//! Available -> Rendering -> Queued -> Presenting -> Available
//!     ^                         |                      |
//!     |                         v                      |
//!     `--------- (dropped) -----'                      |
//!     `-----------------------------------------------|
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::presentation::triple_buffer::{TripleBufferManager, BufferStrategy};
//!
//! // Create a triple buffer manager
//! let mut manager = TripleBufferManager::new(BufferStrategy::Triple);
//!
//! // Acquire a buffer for rendering
//! if let Some(buffer_idx) = manager.acquire_render_buffer() {
//!     // Render to the buffer...
//!
//!     // Submit for presentation
//!     manager.submit_for_present(buffer_idx);
//! }
//!
//! // Present the next queued buffer
//! if let Some(present_idx) = manager.get_present_buffer() {
//!     // Present to display...
//!
//!     // Release after presentation
//!     manager.release_buffer(present_idx);
//! }
//! ```

use std::fmt;
use std::time::{Duration, Instant};

// ============================================================================
// BufferStrategy
// ============================================================================

/// Buffer strategy for presentation.
///
/// Defines how many buffers are used in the swapchain. More buffers reduce
/// GPU stalls but increase latency.
///
/// # Strategies
///
/// - **Double**: 2 buffers - minimal latency, may stall GPU
/// - **Triple**: 3 buffers - balanced latency and throughput (recommended)
/// - **Quadruple**: 4 buffers - maximum throughput, higher latency
/// - **Custom**: N buffers - for specialized use cases
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferStrategy {
    /// Double buffering - 2 buffers.
    ///
    /// One buffer is displayed while the other is rendered to.
    /// Provides the lowest latency but may cause GPU stalls if
    /// the display isn't ready for the next frame.
    Double,

    /// Triple buffering - 3 buffers.
    ///
    /// Adds a queue buffer between rendering and display, allowing
    /// the GPU to continue rendering while waiting for VSync.
    /// Recommended for most applications.
    Triple,

    /// Quadruple buffering - 4 buffers.
    ///
    /// Very smooth presentation with minimal GPU stalls, at the cost
    /// of increased frame latency. Useful for VR or high-refresh
    /// displays where consistent frame timing is critical.
    Quadruple,

    /// Custom buffer count.
    ///
    /// For specialized use cases requiring specific buffer counts.
    /// Values are clamped to a minimum of 1.
    Custom(u32),
}

impl BufferStrategy {
    /// Returns the number of buffers for this strategy.
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert_eq!(BufferStrategy::Double.buffer_count(), 2);
    /// assert_eq!(BufferStrategy::Triple.buffer_count(), 3);
    /// assert_eq!(BufferStrategy::Custom(5).buffer_count(), 5);
    /// ```
    pub const fn buffer_count(&self) -> u32 {
        match self {
            BufferStrategy::Double => 2,
            BufferStrategy::Triple => 3,
            BufferStrategy::Quadruple => 4,
            BufferStrategy::Custom(n) => {
                if *n < 1 { 1 } else { *n }
            }
        }
    }

    /// Returns the typical latency in frames for this strategy.
    ///
    /// This is the number of frames of input lag introduced by
    /// the buffering strategy. Actual latency depends on frame rate
    /// and display timing.
    ///
    /// # Returns
    ///
    /// - Double: 1 frame latency
    /// - Triple: 2 frames latency
    /// - Quadruple: 3 frames latency
    /// - Custom(n): n-1 frames latency
    pub const fn latency_frames(&self) -> u32 {
        let count = self.buffer_count();
        if count > 0 { count - 1 } else { 0 }
    }

    /// Returns a human-readable description of this strategy.
    ///
    /// Includes buffer count and typical use case.
    pub const fn description(&self) -> &'static str {
        match self {
            BufferStrategy::Double => "Double buffering (2 buffers): minimal latency, may stall",
            BufferStrategy::Triple => "Triple buffering (3 buffers): balanced latency and throughput",
            BufferStrategy::Quadruple => "Quadruple buffering (4 buffers): smooth, higher latency",
            BufferStrategy::Custom(_) => "Custom buffering: specialized configuration",
        }
    }

    /// Returns the recommended strategy for VSync operation.
    ///
    /// Triple buffering is recommended for VSync as it prevents GPU
    /// stalls while waiting for vertical blank, without excessive latency.
    pub const fn recommended_for_vsync() -> Self {
        BufferStrategy::Triple
    }

    /// Returns the recommended strategy for low-latency operation.
    ///
    /// Double buffering provides the lowest latency, suitable for
    /// competitive gaming or VSync-off scenarios.
    pub const fn recommended_for_low_latency() -> Self {
        BufferStrategy::Double
    }

    /// Returns true if this strategy prioritizes latency over throughput.
    pub const fn is_low_latency(&self) -> bool {
        matches!(self, BufferStrategy::Double)
    }

    /// Returns true if this strategy prioritizes throughput over latency.
    pub const fn is_high_throughput(&self) -> bool {
        matches!(self, BufferStrategy::Triple | BufferStrategy::Quadruple)
    }

    /// Returns the maximum queue depth for this strategy.
    ///
    /// This is the number of frames that can be queued for presentation
    /// beyond the one currently being displayed.
    pub const fn max_queue_depth(&self) -> u32 {
        let count = self.buffer_count();
        if count > 1 { count - 1 } else { 0 }
    }
}

impl Default for BufferStrategy {
    fn default() -> Self {
        BufferStrategy::Double
    }
}

impl fmt::Display for BufferStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BufferStrategy::Double => write!(f, "Double (2)"),
            BufferStrategy::Triple => write!(f, "Triple (3)"),
            BufferStrategy::Quadruple => write!(f, "Quadruple (4)"),
            BufferStrategy::Custom(n) => write!(f, "Custom ({})", n),
        }
    }
}

// ============================================================================
// BufferState
// ============================================================================

/// State of a buffer in the presentation chain.
///
/// Buffers transition through states as they are used:
///
/// ```text
/// Available -> Rendering -> Queued -> Presenting -> Available
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferState {
    /// Buffer is available for rendering.
    ///
    /// No operations are in progress on this buffer and it can be
    /// acquired for the next frame.
    Available,

    /// Buffer is currently being rendered to.
    ///
    /// The GPU is actively writing to this buffer. It cannot be
    /// used for presentation until rendering completes.
    Rendering,

    /// Buffer is being presented to the display.
    ///
    /// The display is reading from this buffer. It cannot be
    /// reused until the display moves to the next buffer.
    Presenting,

    /// Buffer is queued for presentation.
    ///
    /// Rendering is complete and the buffer is waiting in the
    /// queue to be presented. Used for triple+ buffering to
    /// decouple GPU rendering from display timing.
    Queued,
}

impl BufferState {
    /// Returns true if this buffer can be acquired for rendering.
    pub const fn is_available(&self) -> bool {
        matches!(self, BufferState::Available)
    }

    /// Returns true if this buffer is in use (not available).
    pub const fn is_in_use(&self) -> bool {
        !self.is_available()
    }

    /// Returns true if this buffer can be presented.
    pub const fn can_present(&self) -> bool {
        matches!(self, BufferState::Queued)
    }

    /// Returns a human-readable name for this state.
    pub const fn name(&self) -> &'static str {
        match self {
            BufferState::Available => "Available",
            BufferState::Rendering => "Rendering",
            BufferState::Presenting => "Presenting",
            BufferState::Queued => "Queued",
        }
    }
}

impl Default for BufferState {
    fn default() -> Self {
        BufferState::Available
    }
}

impl fmt::Display for BufferState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// BufferSlot
// ============================================================================

/// A single buffer slot in the presentation chain.
///
/// Tracks the state and metadata of an individual buffer, including
/// timing information for diagnostics.
#[derive(Clone, Debug)]
pub struct BufferSlot {
    /// Index of this buffer in the chain.
    index: u32,

    /// Current state of the buffer.
    state: BufferState,

    /// When this buffer was last used.
    last_used: Option<Instant>,

    /// Frame number that was rendered to this buffer.
    frame_number: u64,
}

impl BufferSlot {
    /// Creates a new buffer slot with the given index.
    ///
    /// The buffer starts in the Available state with no frame assigned.
    pub fn new(index: u32) -> Self {
        Self {
            index,
            state: BufferState::Available,
            last_used: None,
            frame_number: 0,
        }
    }

    /// Returns the index of this buffer.
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Returns the current state of the buffer.
    pub const fn state(&self) -> BufferState {
        self.state
    }

    /// Returns true if this buffer is available for rendering.
    pub const fn is_available(&self) -> bool {
        self.state.is_available()
    }

    /// Returns true if this buffer is currently being rendered to.
    pub const fn is_rendering(&self) -> bool {
        matches!(self.state, BufferState::Rendering)
    }

    /// Returns true if this buffer is queued for presentation.
    pub const fn is_queued(&self) -> bool {
        matches!(self.state, BufferState::Queued)
    }

    /// Returns true if this buffer is being presented.
    pub const fn is_presenting(&self) -> bool {
        matches!(self.state, BufferState::Presenting)
    }

    /// Returns when this buffer was last used.
    pub fn last_used(&self) -> Option<Instant> {
        self.last_used
    }

    /// Returns the frame number currently in this buffer.
    pub const fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Checks if this buffer is stale (hasn't been used recently).
    ///
    /// A buffer is considered stale if it hasn't been used within
    /// the given threshold duration. Stale buffers in the Available
    /// state may indicate rendering pipeline issues.
    pub fn is_stale(&self, threshold: Duration) -> bool {
        match self.last_used {
            Some(t) => t.elapsed() > threshold,
            None => true, // Never used is considered stale
        }
    }

    /// Returns the time since this buffer was last used.
    pub fn time_since_use(&self) -> Option<Duration> {
        self.last_used.map(|t| t.elapsed())
    }

    /// Marks the buffer as currently being rendered to.
    ///
    /// Updates the state to Rendering and records the frame number.
    /// The timestamp is updated to track when rendering started.
    pub fn mark_rendering(&mut self, frame: u64) {
        self.state = BufferState::Rendering;
        self.frame_number = frame;
        self.last_used = Some(Instant::now());
    }

    /// Marks the buffer as queued for presentation.
    ///
    /// Call this after rendering is complete but before the buffer
    /// is presented. Used for triple+ buffering.
    pub fn mark_queued(&mut self) {
        self.state = BufferState::Queued;
        self.last_used = Some(Instant::now());
    }

    /// Marks the buffer as currently being presented.
    ///
    /// Call this when the buffer is being displayed.
    pub fn mark_presenting(&mut self) {
        self.state = BufferState::Presenting;
        self.last_used = Some(Instant::now());
    }

    /// Marks the buffer as available for reuse.
    ///
    /// Call this after the buffer is no longer being presented.
    pub fn mark_available(&mut self) {
        self.state = BufferState::Available;
        self.last_used = Some(Instant::now());
    }

    /// Resets the buffer to its initial state.
    pub fn reset(&mut self) {
        self.state = BufferState::Available;
        self.last_used = None;
        self.frame_number = 0;
    }
}

impl Default for BufferSlot {
    fn default() -> Self {
        Self::new(0)
    }
}

// ============================================================================
// BufferStats
// ============================================================================

/// Statistics about buffer usage.
///
/// Provides diagnostic information about the buffering system,
/// useful for performance tuning and debugging.
#[derive(Clone, Debug, Default)]
pub struct BufferStats {
    /// Total number of buffers in the chain.
    pub buffer_count: u32,

    /// Number of buffers currently available.
    pub available_count: u32,

    /// Number of buffers currently queued for presentation.
    pub queued_count: u32,

    /// Number of buffers currently being rendered to.
    pub rendering_count: u32,

    /// Number of buffers currently being presented.
    pub presenting_count: u32,

    /// Total frames that have been dropped due to queue overflow.
    pub frames_dropped: u64,

    /// Average queue depth over recent frames.
    pub avg_queue_depth: f64,

    /// Total frames rendered.
    pub total_frames: u64,

    /// Number of times no buffer was available (stalls).
    pub stall_count: u64,
}

impl BufferStats {
    /// Returns the utilization ratio (0.0 - 1.0).
    ///
    /// Higher values indicate more buffers are in use.
    pub fn utilization(&self) -> f64 {
        if self.buffer_count == 0 {
            return 0.0;
        }
        let in_use = self.buffer_count - self.available_count;
        in_use as f64 / self.buffer_count as f64
    }

    /// Returns true if there are no available buffers.
    pub fn is_starved(&self) -> bool {
        self.available_count == 0
    }

    /// Returns true if all buffers are available (idle).
    pub fn is_idle(&self) -> bool {
        self.available_count == self.buffer_count
    }

    /// Returns the drop rate as a ratio (0.0 - 1.0).
    pub fn drop_rate(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.frames_dropped as f64 / self.total_frames as f64
    }

    /// Returns the stall rate as a ratio (0.0 - 1.0).
    pub fn stall_rate(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.stall_count as f64 / self.total_frames as f64
    }
}

impl fmt::Display for BufferStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Buffers: {}/{} available, {} queued, {} dropped ({:.1}% util)",
            self.available_count,
            self.buffer_count,
            self.queued_count,
            self.frames_dropped,
            self.utilization() * 100.0
        )
    }
}

// ============================================================================
// TripleBufferManager
// ============================================================================

/// Manager for triple (and multi-) buffering in the presentation system.
///
/// Handles buffer acquisition, submission, and release in a thread-safe
/// manner. Supports various buffering strategies from double to custom.
///
/// # Thread Safety
///
/// The manager itself is not thread-safe; synchronize access externally
/// if used from multiple threads.
///
/// # Example
///
/// ```ignore
/// let mut manager = TripleBufferManager::new(BufferStrategy::Triple);
///
/// // Render loop
/// loop {
///     // Acquire buffer for rendering
///     let buffer = manager.acquire_render_buffer().unwrap();
///
///     // ... render to buffer ...
///
///     // Submit for presentation
///     manager.submit_for_present(buffer);
///
///     // Get next buffer to present
///     if let Some(present) = manager.get_present_buffer() {
///         // ... present to display ...
///         manager.release_buffer(present);
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct TripleBufferManager {
    /// The buffering strategy in use.
    strategy: BufferStrategy,

    /// Buffer slots in the chain.
    buffers: Vec<BufferSlot>,

    /// Index of the buffer currently being rendered to.
    current_render_index: Option<u32>,

    /// Index of the buffer currently being presented.
    present_index: Option<u32>,

    /// Frame counter for tracking frame numbers.
    frame_counter: u64,

    /// Count of dropped frames.
    frames_dropped: u64,

    /// Count of stalls (no buffer available).
    stall_count: u64,

    /// Queue depth history for averaging.
    queue_depth_history: Vec<u32>,

    /// Maximum history entries for averaging.
    max_history: usize,
}

impl TripleBufferManager {
    /// Creates a new triple buffer manager with the given strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The buffering strategy to use.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let manager = TripleBufferManager::new(BufferStrategy::Triple);
    /// assert_eq!(manager.buffer_count(), 3);
    /// ```
    pub fn new(strategy: BufferStrategy) -> Self {
        let count = strategy.buffer_count();
        let buffers = (0..count).map(BufferSlot::new).collect();

        Self {
            strategy,
            buffers,
            current_render_index: None,
            present_index: None,
            frame_counter: 0,
            frames_dropped: 0,
            stall_count: 0,
            queue_depth_history: Vec::with_capacity(60),
            max_history: 60,
        }
    }

    /// Creates a new manager optimized for VSync.
    pub fn new_vsync() -> Self {
        Self::new(BufferStrategy::recommended_for_vsync())
    }

    /// Creates a new manager optimized for low latency.
    pub fn new_low_latency() -> Self {
        Self::new(BufferStrategy::recommended_for_low_latency())
    }

    /// Returns the current buffering strategy.
    pub const fn strategy(&self) -> BufferStrategy {
        self.strategy
    }

    /// Returns the number of buffers in the chain.
    pub fn buffer_count(&self) -> u32 {
        self.buffers.len() as u32
    }

    /// Returns the number of currently available buffers.
    pub fn available_buffers(&self) -> u32 {
        self.buffers.iter().filter(|b| b.is_available()).count() as u32
    }

    /// Returns the number of buffers queued for presentation.
    pub fn queued_buffers(&self) -> u32 {
        self.buffers.iter().filter(|b| b.is_queued()).count() as u32
    }

    /// Returns the number of buffers currently being rendered to.
    pub fn rendering_buffers(&self) -> u32 {
        self.buffers.iter().filter(|b| b.is_rendering()).count() as u32
    }

    /// Returns the current frame number.
    pub const fn frame_number(&self) -> u64 {
        self.frame_counter
    }

    /// Acquires a buffer for rendering.
    ///
    /// Returns the index of an available buffer, or `None` if no buffers
    /// are available. The buffer is marked as Rendering.
    ///
    /// # Returns
    ///
    /// - `Some(index)` - Index of the acquired buffer
    /// - `None` - No buffer available (all in use)
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(buffer_idx) = manager.acquire_render_buffer() {
    ///     // Render to buffer at index buffer_idx
    /// } else {
    ///     // No buffer available, skip frame or wait
    /// }
    /// ```
    pub fn acquire_render_buffer(&mut self) -> Option<u32> {
        // Find an available buffer, preferring the one least recently used
        let available_idx = self.buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| b.is_available())
            .min_by(|(_, a), (_, b)| {
                match (a.last_used(), b.last_used()) {
                    (None, None) => std::cmp::Ordering::Equal,
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (Some(a_time), Some(b_time)) => a_time.cmp(&b_time),
                }
            })
            .map(|(i, _)| i);

        match available_idx {
            Some(idx) => {
                self.frame_counter += 1;
                self.buffers[idx].mark_rendering(self.frame_counter);
                self.current_render_index = Some(idx as u32);
                Some(idx as u32)
            }
            None => {
                self.stall_count += 1;
                None
            }
        }
    }

    /// Submits a rendered buffer for presentation.
    ///
    /// Marks the buffer as Queued. If the queue is full (all buffers except
    /// the presenting one are queued), the oldest queued buffer is dropped.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the buffer to submit
    ///
    /// # Returns
    ///
    /// `true` if the buffer was submitted successfully, `false` if the index
    /// was invalid or the buffer wasn't in the Rendering state.
    pub fn submit_for_present(&mut self, index: u32) -> bool {
        let idx = index as usize;
        if idx >= self.buffers.len() {
            return false;
        }

        if !self.buffers[idx].is_rendering() {
            return false;
        }

        // Check queue depth and drop oldest if needed
        let max_queued = self.strategy.max_queue_depth().saturating_sub(1);
        let current_queued = self.queued_buffers();

        if current_queued >= max_queued && max_queued > 0 {
            // Find and drop oldest queued buffer
            if let Some(oldest_idx) = self.find_oldest_queued() {
                self.buffers[oldest_idx].mark_available();
                self.frames_dropped += 1;
            }
        }

        self.buffers[idx].mark_queued();

        if self.current_render_index == Some(index) {
            self.current_render_index = None;
        }

        // Update queue depth history
        self.update_queue_depth_history();

        true
    }

    /// Gets the next buffer to present.
    ///
    /// Returns the index of the oldest queued buffer, or `None` if no
    /// buffers are queued. The buffer is marked as Presenting.
    ///
    /// # Returns
    ///
    /// - `Some(index)` - Index of the buffer to present
    /// - `None` - No buffers queued for presentation
    pub fn get_present_buffer(&mut self) -> Option<u32> {
        // Find the oldest queued buffer (lowest frame number)
        let queued_idx = self.buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| b.is_queued())
            .min_by_key(|(_, b)| b.frame_number())
            .map(|(i, _)| i);

        if let Some(idx) = queued_idx {
            // Release the previously presenting buffer
            if let Some(prev_idx) = self.present_index {
                let prev = prev_idx as usize;
                if prev < self.buffers.len() && self.buffers[prev].is_presenting() {
                    self.buffers[prev].mark_available();
                }
            }

            self.buffers[idx].mark_presenting();
            self.present_index = Some(idx as u32);
            Some(idx as u32)
        } else {
            None
        }
    }

    /// Releases a buffer after presentation.
    ///
    /// Marks the buffer as Available for reuse.
    ///
    /// # Arguments
    ///
    /// * `index` - Index of the buffer to release
    ///
    /// # Returns
    ///
    /// `true` if the buffer was released, `false` if invalid.
    pub fn release_buffer(&mut self, index: u32) -> bool {
        let idx = index as usize;
        if idx >= self.buffers.len() {
            return false;
        }

        self.buffers[idx].mark_available();

        if self.present_index == Some(index) {
            self.present_index = None;
        }

        true
    }

    /// Returns statistics about buffer usage.
    pub fn stats(&self) -> BufferStats {
        let avg_queue_depth = if self.queue_depth_history.is_empty() {
            0.0
        } else {
            let sum: u32 = self.queue_depth_history.iter().sum();
            sum as f64 / self.queue_depth_history.len() as f64
        };

        BufferStats {
            buffer_count: self.buffer_count(),
            available_count: self.available_buffers(),
            queued_count: self.queued_buffers(),
            rendering_count: self.rendering_buffers(),
            presenting_count: self.buffers.iter()
                .filter(|b| b.is_presenting())
                .count() as u32,
            frames_dropped: self.frames_dropped,
            avg_queue_depth,
            total_frames: self.frame_counter,
            stall_count: self.stall_count,
        }
    }

    /// Resets the manager to its initial state.
    ///
    /// All buffers are marked as Available and counters are reset.
    pub fn reset(&mut self) {
        for buffer in &mut self.buffers {
            buffer.reset();
        }
        self.current_render_index = None;
        self.present_index = None;
        self.frame_counter = 0;
        self.frames_dropped = 0;
        self.stall_count = 0;
        self.queue_depth_history.clear();
    }

    /// Resizes the manager to use a new buffering strategy.
    ///
    /// All in-flight frames are lost. Call during idle periods.
    pub fn resize(&mut self, strategy: BufferStrategy) {
        let count = strategy.buffer_count();
        self.strategy = strategy;
        self.buffers = (0..count).map(BufferSlot::new).collect();
        self.current_render_index = None;
        self.present_index = None;
        // Keep frame counter and stats for continuity
    }

    /// Returns a reference to a buffer slot by index.
    pub fn get_buffer(&self, index: u32) -> Option<&BufferSlot> {
        self.buffers.get(index as usize)
    }

    /// Returns the index of the currently rendering buffer.
    pub fn current_render_index(&self) -> Option<u32> {
        self.current_render_index
    }

    /// Returns the index of the currently presenting buffer.
    pub fn present_index(&self) -> Option<u32> {
        self.present_index
    }

    /// Checks if any buffers are stale.
    pub fn has_stale_buffers(&self, threshold: Duration) -> bool {
        self.buffers.iter().any(|b| b.is_available() && b.is_stale(threshold))
    }

    /// Returns the number of stale buffers.
    pub fn stale_buffer_count(&self, threshold: Duration) -> u32 {
        self.buffers
            .iter()
            .filter(|b| b.is_available() && b.is_stale(threshold))
            .count() as u32
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    fn find_oldest_queued(&self) -> Option<usize> {
        self.buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| b.is_queued())
            .min_by_key(|(_, b)| b.frame_number())
            .map(|(i, _)| i)
    }

    fn update_queue_depth_history(&mut self) {
        let depth = self.queued_buffers();
        self.queue_depth_history.push(depth);
        if self.queue_depth_history.len() > self.max_history {
            self.queue_depth_history.remove(0);
        }
    }
}

impl Default for TripleBufferManager {
    fn default() -> Self {
        Self::new(BufferStrategy::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BufferStrategy tests
    // ========================================================================

    #[test]
    fn test_buffer_strategy_double_count() {
        assert_eq!(BufferStrategy::Double.buffer_count(), 2);
    }

    #[test]
    fn test_buffer_strategy_triple_count() {
        assert_eq!(BufferStrategy::Triple.buffer_count(), 3);
    }

    #[test]
    fn test_buffer_strategy_quadruple_count() {
        assert_eq!(BufferStrategy::Quadruple.buffer_count(), 4);
    }

    #[test]
    fn test_buffer_strategy_custom_count() {
        assert_eq!(BufferStrategy::Custom(5).buffer_count(), 5);
        assert_eq!(BufferStrategy::Custom(1).buffer_count(), 1);
        assert_eq!(BufferStrategy::Custom(0).buffer_count(), 1); // Clamped
    }

    #[test]
    fn test_buffer_strategy_latency_frames_double() {
        assert_eq!(BufferStrategy::Double.latency_frames(), 1);
    }

    #[test]
    fn test_buffer_strategy_latency_frames_triple() {
        assert_eq!(BufferStrategy::Triple.latency_frames(), 2);
    }

    #[test]
    fn test_buffer_strategy_latency_frames_quadruple() {
        assert_eq!(BufferStrategy::Quadruple.latency_frames(), 3);
    }

    #[test]
    fn test_buffer_strategy_latency_frames_custom() {
        assert_eq!(BufferStrategy::Custom(5).latency_frames(), 4);
        assert_eq!(BufferStrategy::Custom(1).latency_frames(), 0);
    }

    #[test]
    fn test_buffer_strategy_descriptions() {
        assert!(!BufferStrategy::Double.description().is_empty());
        assert!(!BufferStrategy::Triple.description().is_empty());
        assert!(!BufferStrategy::Quadruple.description().is_empty());
        assert!(!BufferStrategy::Custom(3).description().is_empty());
    }

    #[test]
    fn test_buffer_strategy_recommendations() {
        assert_eq!(BufferStrategy::recommended_for_vsync(), BufferStrategy::Triple);
        assert_eq!(BufferStrategy::recommended_for_low_latency(), BufferStrategy::Double);
    }

    #[test]
    fn test_buffer_strategy_default() {
        assert_eq!(BufferStrategy::default(), BufferStrategy::Double);
    }

    #[test]
    fn test_buffer_strategy_is_low_latency() {
        assert!(BufferStrategy::Double.is_low_latency());
        assert!(!BufferStrategy::Triple.is_low_latency());
        assert!(!BufferStrategy::Quadruple.is_low_latency());
    }

    #[test]
    fn test_buffer_strategy_is_high_throughput() {
        assert!(!BufferStrategy::Double.is_high_throughput());
        assert!(BufferStrategy::Triple.is_high_throughput());
        assert!(BufferStrategy::Quadruple.is_high_throughput());
    }

    #[test]
    fn test_buffer_strategy_max_queue_depth() {
        assert_eq!(BufferStrategy::Double.max_queue_depth(), 1);
        assert_eq!(BufferStrategy::Triple.max_queue_depth(), 2);
        assert_eq!(BufferStrategy::Quadruple.max_queue_depth(), 3);
    }

    #[test]
    fn test_buffer_strategy_display() {
        assert_eq!(format!("{}", BufferStrategy::Double), "Double (2)");
        assert_eq!(format!("{}", BufferStrategy::Triple), "Triple (3)");
        assert_eq!(format!("{}", BufferStrategy::Custom(7)), "Custom (7)");
    }

    // ========================================================================
    // BufferState tests
    // ========================================================================

    #[test]
    fn test_buffer_state_is_available() {
        assert!(BufferState::Available.is_available());
        assert!(!BufferState::Rendering.is_available());
        assert!(!BufferState::Presenting.is_available());
        assert!(!BufferState::Queued.is_available());
    }

    #[test]
    fn test_buffer_state_is_in_use() {
        assert!(!BufferState::Available.is_in_use());
        assert!(BufferState::Rendering.is_in_use());
        assert!(BufferState::Presenting.is_in_use());
        assert!(BufferState::Queued.is_in_use());
    }

    #[test]
    fn test_buffer_state_can_present() {
        assert!(!BufferState::Available.can_present());
        assert!(!BufferState::Rendering.can_present());
        assert!(!BufferState::Presenting.can_present());
        assert!(BufferState::Queued.can_present());
    }

    #[test]
    fn test_buffer_state_names() {
        assert_eq!(BufferState::Available.name(), "Available");
        assert_eq!(BufferState::Rendering.name(), "Rendering");
        assert_eq!(BufferState::Presenting.name(), "Presenting");
        assert_eq!(BufferState::Queued.name(), "Queued");
    }

    #[test]
    fn test_buffer_state_default() {
        assert_eq!(BufferState::default(), BufferState::Available);
    }

    // ========================================================================
    // BufferSlot tests
    // ========================================================================

    #[test]
    fn test_buffer_slot_new() {
        let slot = BufferSlot::new(5);
        assert_eq!(slot.index(), 5);
        assert_eq!(slot.state(), BufferState::Available);
        assert!(slot.is_available());
        assert_eq!(slot.frame_number(), 0);
        assert!(slot.last_used().is_none());
    }

    #[test]
    fn test_buffer_slot_state_transitions() {
        let mut slot = BufferSlot::new(0);

        // Start available
        assert!(slot.is_available());

        // Mark rendering
        slot.mark_rendering(42);
        assert!(slot.is_rendering());
        assert_eq!(slot.frame_number(), 42);
        assert!(slot.last_used().is_some());

        // Mark queued
        slot.mark_queued();
        assert!(slot.is_queued());

        // Mark presenting
        slot.mark_presenting();
        assert!(slot.is_presenting());

        // Mark available
        slot.mark_available();
        assert!(slot.is_available());
    }

    #[test]
    fn test_buffer_slot_staleness() {
        let mut slot = BufferSlot::new(0);

        // Never used is stale
        assert!(slot.is_stale(Duration::from_secs(0)));

        // Recently used is not stale
        slot.mark_available();
        assert!(!slot.is_stale(Duration::from_secs(10)));

        // Very short threshold
        assert!(!slot.is_stale(Duration::from_millis(100)));
    }

    #[test]
    fn test_buffer_slot_reset() {
        let mut slot = BufferSlot::new(3);
        slot.mark_rendering(100);
        slot.mark_queued();

        slot.reset();

        assert!(slot.is_available());
        assert_eq!(slot.frame_number(), 0);
        assert!(slot.last_used().is_none());
    }

    #[test]
    fn test_buffer_slot_time_since_use() {
        let mut slot = BufferSlot::new(0);

        assert!(slot.time_since_use().is_none());

        slot.mark_available();
        let elapsed = slot.time_since_use();
        assert!(elapsed.is_some());
        assert!(elapsed.unwrap() < Duration::from_secs(1));
    }

    // ========================================================================
    // BufferStats tests
    // ========================================================================

    #[test]
    fn test_buffer_stats_utilization() {
        let stats = BufferStats {
            buffer_count: 3,
            available_count: 1,
            ..Default::default()
        };

        let util = stats.utilization();
        assert!((util - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_buffer_stats_is_starved() {
        let stats = BufferStats {
            buffer_count: 3,
            available_count: 0,
            ..Default::default()
        };
        assert!(stats.is_starved());

        let stats2 = BufferStats {
            buffer_count: 3,
            available_count: 1,
            ..Default::default()
        };
        assert!(!stats2.is_starved());
    }

    #[test]
    fn test_buffer_stats_is_idle() {
        let stats = BufferStats {
            buffer_count: 3,
            available_count: 3,
            ..Default::default()
        };
        assert!(stats.is_idle());

        let stats2 = BufferStats {
            buffer_count: 3,
            available_count: 2,
            ..Default::default()
        };
        assert!(!stats2.is_idle());
    }

    #[test]
    fn test_buffer_stats_drop_rate() {
        let stats = BufferStats {
            total_frames: 100,
            frames_dropped: 5,
            ..Default::default()
        };
        assert!((stats.drop_rate() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_buffer_stats_stall_rate() {
        let stats = BufferStats {
            total_frames: 100,
            stall_count: 10,
            ..Default::default()
        };
        assert!((stats.stall_rate() - 0.10).abs() < 0.001);
    }

    // ========================================================================
    // TripleBufferManager tests
    // ========================================================================

    #[test]
    fn test_manager_initialization_double() {
        let manager = TripleBufferManager::new(BufferStrategy::Double);
        assert_eq!(manager.buffer_count(), 2);
        assert_eq!(manager.strategy(), BufferStrategy::Double);
        assert_eq!(manager.available_buffers(), 2);
    }

    #[test]
    fn test_manager_initialization_triple() {
        let manager = TripleBufferManager::new(BufferStrategy::Triple);
        assert_eq!(manager.buffer_count(), 3);
        assert_eq!(manager.strategy(), BufferStrategy::Triple);
        assert_eq!(manager.available_buffers(), 3);
    }

    #[test]
    fn test_manager_initialization_quadruple() {
        let manager = TripleBufferManager::new(BufferStrategy::Quadruple);
        assert_eq!(manager.buffer_count(), 4);
        assert_eq!(manager.available_buffers(), 4);
    }

    #[test]
    fn test_manager_initialization_custom() {
        let manager = TripleBufferManager::new(BufferStrategy::Custom(6));
        assert_eq!(manager.buffer_count(), 6);
        assert_eq!(manager.available_buffers(), 6);
    }

    #[test]
    fn test_manager_acquire_render_buffer() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        let buffer = manager.acquire_render_buffer();
        assert!(buffer.is_some());
        assert_eq!(manager.available_buffers(), 2);
        assert_eq!(manager.rendering_buffers(), 1);
        assert_eq!(manager.frame_number(), 1);
    }

    #[test]
    fn test_manager_acquire_all_buffers() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        // With triple buffering: one presenting, one queued, one rendering
        // Acquire first buffer
        let buf0 = manager.acquire_render_buffer().unwrap();
        assert_eq!(manager.available_buffers(), 2);
        assert_eq!(manager.rendering_buffers(), 1);

        // Submit first buffer for presentation
        manager.submit_for_present(buf0);
        assert_eq!(manager.queued_buffers(), 1);

        // Acquire second buffer
        let buf1 = manager.acquire_render_buffer().unwrap();
        assert_eq!(manager.available_buffers(), 1);
        assert_eq!(manager.rendering_buffers(), 1);

        // Now present the first buffer
        let present = manager.get_present_buffer().unwrap();
        assert_eq!(present, buf0);

        // Submit second buffer
        manager.submit_for_present(buf1);

        // Acquire third buffer
        let _buf2 = manager.acquire_render_buffer().unwrap();

        // Now: one presenting (buf0), one queued (buf1), one rendering (buf2)
        assert_eq!(manager.available_buffers(), 0);
    }

    #[test]
    fn test_manager_submit_for_present() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        let buffer = manager.acquire_render_buffer().unwrap();
        assert!(manager.submit_for_present(buffer));

        assert_eq!(manager.queued_buffers(), 1);
        assert_eq!(manager.rendering_buffers(), 0);
    }

    #[test]
    fn test_manager_get_present_buffer() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        // Queue a buffer
        let buffer = manager.acquire_render_buffer().unwrap();
        manager.submit_for_present(buffer);

        // Get for presentation
        let present = manager.get_present_buffer();
        assert!(present.is_some());
        assert_eq!(present.unwrap(), buffer);
        assert_eq!(manager.queued_buffers(), 0);
    }

    #[test]
    fn test_manager_release_buffer() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        let buffer = manager.acquire_render_buffer().unwrap();
        manager.submit_for_present(buffer);
        manager.get_present_buffer();

        assert!(manager.release_buffer(buffer));
        assert_eq!(manager.available_buffers(), 3);
    }

    #[test]
    fn test_manager_buffer_cycling() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        // Simulate several frames
        for i in 0..5 {
            let buffer = manager.acquire_render_buffer();
            assert!(buffer.is_some(), "Frame {} acquire failed", i);

            let idx = buffer.unwrap();
            assert!(manager.submit_for_present(idx));

            if let Some(present) = manager.get_present_buffer() {
                manager.release_buffer(present);
            }
        }

        // Manager should still be functional
        assert!(manager.acquire_render_buffer().is_some());
    }

    #[test]
    fn test_manager_stats() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        // Acquire and queue a buffer
        let buffer = manager.acquire_render_buffer().unwrap();
        manager.submit_for_present(buffer);

        let stats = manager.stats();
        assert_eq!(stats.buffer_count, 3);
        assert_eq!(stats.queued_count, 1);
        assert_eq!(stats.total_frames, 1);
    }

    #[test]
    fn test_manager_reset() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        // Use the manager
        let buffer = manager.acquire_render_buffer().unwrap();
        manager.submit_for_present(buffer);

        // Reset
        manager.reset();

        assert_eq!(manager.frame_number(), 0);
        assert_eq!(manager.available_buffers(), 3);
        assert_eq!(manager.queued_buffers(), 0);
    }

    #[test]
    fn test_manager_resize() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Double);
        assert_eq!(manager.buffer_count(), 2);

        manager.resize(BufferStrategy::Quadruple);
        assert_eq!(manager.buffer_count(), 4);
        assert_eq!(manager.strategy(), BufferStrategy::Quadruple);
        assert_eq!(manager.available_buffers(), 4);
    }

    #[test]
    fn test_manager_stall_count() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Double);

        // Acquire both buffers
        let b1 = manager.acquire_render_buffer().unwrap();
        let b2 = manager.acquire_render_buffer().unwrap();

        // Try to acquire when none available
        let result = manager.acquire_render_buffer();
        assert!(result.is_none());

        let stats = manager.stats();
        assert_eq!(stats.stall_count, 1);

        // Release and acquire again
        manager.submit_for_present(b1);
        manager.release_buffer(b1);
        assert!(manager.acquire_render_buffer().is_some());
    }

    #[test]
    fn test_manager_new_vsync() {
        let manager = TripleBufferManager::new_vsync();
        assert_eq!(manager.strategy(), BufferStrategy::Triple);
    }

    #[test]
    fn test_manager_new_low_latency() {
        let manager = TripleBufferManager::new_low_latency();
        assert_eq!(manager.strategy(), BufferStrategy::Double);
    }

    #[test]
    fn test_manager_get_buffer() {
        let manager = TripleBufferManager::new(BufferStrategy::Triple);

        assert!(manager.get_buffer(0).is_some());
        assert!(manager.get_buffer(1).is_some());
        assert!(manager.get_buffer(2).is_some());
        assert!(manager.get_buffer(3).is_none());
    }

    #[test]
    fn test_manager_default() {
        let manager = TripleBufferManager::default();
        assert_eq!(manager.strategy(), BufferStrategy::Double);
    }

    #[test]
    fn test_manager_render_present_indices() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        assert!(manager.current_render_index().is_none());
        assert!(manager.present_index().is_none());

        let buffer = manager.acquire_render_buffer().unwrap();
        assert_eq!(manager.current_render_index(), Some(buffer));

        manager.submit_for_present(buffer);
        assert!(manager.current_render_index().is_none());

        let present = manager.get_present_buffer().unwrap();
        assert_eq!(manager.present_index(), Some(present));
    }

    #[test]
    fn test_manager_stale_buffers() {
        let manager = TripleBufferManager::new(BufferStrategy::Triple);

        // All buffers are stale (never used)
        assert!(manager.has_stale_buffers(Duration::from_secs(0)));
        assert_eq!(manager.stale_buffer_count(Duration::from_secs(0)), 3);
    }

    #[test]
    fn test_manager_invalid_submit() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        // Submit invalid index
        assert!(!manager.submit_for_present(10));

        // Submit without acquiring first
        assert!(!manager.submit_for_present(0));
    }

    #[test]
    fn test_manager_invalid_release() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Triple);

        // Release invalid index
        assert!(!manager.release_buffer(10));
    }

    #[test]
    fn test_manager_frame_ordering() {
        let mut manager = TripleBufferManager::new(BufferStrategy::Quadruple);

        // With quadruple buffering, we can queue more frames
        // Acquire and submit multiple frames
        let buf0 = manager.acquire_render_buffer().unwrap();
        manager.submit_for_present(buf0);

        let buf1 = manager.acquire_render_buffer().unwrap();
        manager.submit_for_present(buf1);

        // Now we have 2 buffers queued
        assert_eq!(manager.queued_buffers(), 2);

        // Present should return oldest first (frame 1)
        let present1 = manager.get_present_buffer().unwrap();
        let frame1 = manager.get_buffer(present1).unwrap().frame_number();
        assert_eq!(frame1, 1);

        // First frame is now presenting, second is still queued
        assert_eq!(manager.queued_buffers(), 1);

        // Present the second frame (frame 2)
        let present2 = manager.get_present_buffer().unwrap();
        let frame2 = manager.get_buffer(present2).unwrap().frame_number();
        assert_eq!(frame2, 2);

        assert!(frame2 > frame1, "Frames should be presented in order");
    }
}
