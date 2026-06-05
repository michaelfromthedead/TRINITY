//! TrinityCommandEncoder - Command encoder wrapper for wgpu 22.x/25.x
//!
//! This module provides the [`TrinityCommandEncoder`] struct, which wraps wgpu's
//! command encoder with additional metadata for frame tracking and debugging.
//!
//! # Overview
//!
//! The command encoder is the primary mechanism for recording GPU commands in wgpu.
//! This wrapper adds:
//!
//! - Frame number tracking for synchronization and profiling
//! - Device reference for creating additional resources during recording
//! - Label support for debugging with RenderDoc and similar tools
//! - State machine for tracking encoder lifecycle
//! - Active pass tracking with error handling
//!
//! # Encoder State Machine
//!
//! The encoder tracks its state through a state machine with the following transitions:
//!
//! ```text
//!     ┌──────────────────────────────────────────────┐
//!     │                                              │
//!     v                                              │
//! [Created] ──begin_render_pass──> [InRenderPass] ──┘
//!     │                                   │
//!     │                              end_render_pass
//!     │                                   │
//!     │                                   v
//!     ├──begin_compute_pass─> [InComputePass] ──────┘
//!     │                              │
//!     │                         end_compute_pass
//!     │                              │
//!     v                              v
//!   finish() ──────────────────> [Finished]
//! ```
//!
//! - **Created**: Initial state, ready to record commands or begin passes
//! - **InRenderPass**: A render pass is active (encoder is borrowed)
//! - **InComputePass**: A compute pass is active (encoder is borrowed)
//! - **Finished**: The encoder has been consumed to produce a CommandBuffer
//!
//! # Pass Tracking
//!
//! The encoder provides safe pass tracking through [`ActivePassType`] and prevents
//! common errors like beginning a new pass while one is active:
//!
//! ```no_run
//! use std::sync::Arc;
//! use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor, ActivePassType};
//!
//! # fn example(device: Arc<wgpu::Device>) {
//! let mut encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
//!
//! assert_eq!(encoder.active_pass(), ActivePassType::None);
//! assert!(!encoder.has_active_pass());
//! # }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor, EncoderState};
//!
//! # fn example(device: Arc<wgpu::Device>) {
//! // Create encoder descriptor with a label
//! let desc = CommandEncoderDescriptor::new()
//!     .label("Main Frame Encoder");
//!
//! // Create the encoder for frame 42
//! let encoder = TrinityCommandEncoder::new(&device, &desc, 42);
//!
//! assert_eq!(encoder.frame_number(), 42);
//! assert_eq!(encoder.label(), Some("Main Frame Encoder"));
//! assert_eq!(encoder.state(), EncoderState::Created);
//! assert!(encoder.is_recording());
//! # }
//! ```
//!
//! # Thread Safety
//!
//! [`TrinityCommandEncoder`] is `Send + Sync` because:
//! - `wgpu::CommandEncoder` is `Send + Sync`
//! - `Arc<wgpu::Device>` is `Send + Sync`
//! - All other fields are primitive types or `String`
//! - State tracking uses `AtomicU8` for thread-safe access
//!
//! However, command encoders should typically be used from a single thread
//! at a time due to the mutable state involved in recording commands.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

// ============================================================================
// EncoderState
// ============================================================================

/// Represents the lifecycle state of a [`TrinityCommandEncoder`].
///
/// The encoder follows a state machine pattern to ensure correct usage:
///
/// - Only one pass (render or compute) can be active at a time
/// - Finish can only be called from the Created state
/// - State transitions are validated in debug builds
///
/// # State Diagram
///
/// ```text
/// Created ─────┬────> InRenderPass ─────┬────> Created
///              │                        │
///              └────> InComputePass ────┘
///              │
///              └────> Finished (terminal)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EncoderState {
    /// Initial state after creation. Ready to:
    /// - Begin a render pass (-> InRenderPass)
    /// - Begin a compute pass (-> InComputePass)
    /// - Record copy/clear commands
    /// - Finish and produce a CommandBuffer (-> Finished)
    Created = 0,

    /// A render pass is currently active.
    /// The encoder is borrowed by the RenderPass object.
    /// Can only transition back to Created when the pass ends.
    InRenderPass = 1,

    /// A compute pass is currently active.
    /// The encoder is borrowed by the ComputePass object.
    /// Can only transition back to Created when the pass ends.
    InComputePass = 2,

    /// The encoder has been consumed via `finish()`.
    /// This is a terminal state - no further operations are valid.
    Finished = 3,
}

impl EncoderState {
    /// Convert from u8 to EncoderState.
    ///
    /// Returns `Created` for any unrecognized value (defensive programming).
    #[inline]
    fn from_u8(value: u8) -> Self {
        match value {
            0 => EncoderState::Created,
            1 => EncoderState::InRenderPass,
            2 => EncoderState::InComputePass,
            3 => EncoderState::Finished,
            _ => EncoderState::Created, // Fallback for safety
        }
    }

    /// Check if this state allows beginning a new pass.
    #[inline]
    pub fn can_begin_pass(self) -> bool {
        matches!(self, EncoderState::Created)
    }

    /// Check if this state allows finishing the encoder.
    #[inline]
    pub fn can_finish(self) -> bool {
        matches!(self, EncoderState::Created)
    }

    /// Check if this state represents an active pass.
    #[inline]
    pub fn is_in_pass(self) -> bool {
        matches!(self, EncoderState::InRenderPass | EncoderState::InComputePass)
    }
}

impl std::fmt::Display for EncoderState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncoderState::Created => write!(f, "Created"),
            EncoderState::InRenderPass => write!(f, "InRenderPass"),
            EncoderState::InComputePass => write!(f, "InComputePass"),
            EncoderState::Finished => write!(f, "Finished"),
        }
    }
}

// ============================================================================
// ActivePassType
// ============================================================================

/// Represents the type of currently active pass in the encoder.
///
/// This enum provides a simplified view of the pass state for external consumers
/// who only need to know if a pass is active and what type it is.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::command_encoder::ActivePassType;
///
/// let pass_type = ActivePassType::None;
/// assert!(!pass_type.is_active());
///
/// let render_pass = ActivePassType::Render;
/// assert!(render_pass.is_active());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivePassType {
    /// No pass is currently active.
    None,
    /// A render pass is currently active.
    Render,
    /// A compute pass is currently active.
    Compute,
}

impl ActivePassType {
    /// Check if there is an active pass.
    ///
    /// Returns `true` if either a render or compute pass is active.
    #[inline]
    pub fn is_active(self) -> bool {
        !matches!(self, ActivePassType::None)
    }

    /// Convert from [`EncoderState`] to [`ActivePassType`].
    ///
    /// Maps `InRenderPass` -> `Render`, `InComputePass` -> `Compute`,
    /// and all other states -> `None`.
    #[inline]
    pub fn from_encoder_state(state: EncoderState) -> Self {
        match state {
            EncoderState::InRenderPass => ActivePassType::Render,
            EncoderState::InComputePass => ActivePassType::Compute,
            _ => ActivePassType::None,
        }
    }
}

impl std::fmt::Display for ActivePassType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivePassType::None => write!(f, "None"),
            ActivePassType::Render => write!(f, "Render"),
            ActivePassType::Compute => write!(f, "Compute"),
        }
    }
}

impl Default for ActivePassType {
    fn default() -> Self {
        ActivePassType::None
    }
}

// ============================================================================
// PassError
// ============================================================================

/// Error type for pass operations.
///
/// Returned when pass operations fail due to invalid state, such as:
/// - Attempting to begin a pass when one is already active
/// - Attempting to end a pass when none is active
/// - Attempting to end the wrong type of pass
///
/// # Example
///
/// ```no_run
/// use renderer_backend::command_encoder::{PassError, PassErrorKind};
///
/// let error = PassError::new(PassErrorKind::PassAlreadyActive);
/// assert_eq!(error.kind, PassErrorKind::PassAlreadyActive);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassError {
    /// The kind of pass error that occurred.
    pub kind: PassErrorKind,
}

impl PassError {
    /// Create a new pass error with the given kind.
    #[inline]
    pub fn new(kind: PassErrorKind) -> Self {
        Self { kind }
    }

    /// Create a `PassAlreadyActive` error.
    #[inline]
    pub fn already_active() -> Self {
        Self::new(PassErrorKind::PassAlreadyActive)
    }

    /// Create a `NoActivePass` error.
    #[inline]
    pub fn no_active() -> Self {
        Self::new(PassErrorKind::NoActivePass)
    }

    /// Create a `WrongPassType` error.
    #[inline]
    pub fn wrong_type() -> Self {
        Self::new(PassErrorKind::WrongPassType)
    }

    /// Create an `EncoderFinished` error.
    #[inline]
    pub fn encoder_finished() -> Self {
        Self::new(PassErrorKind::EncoderFinished)
    }
}

impl std::fmt::Display for PassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for PassError {}

// ============================================================================
// PassErrorKind
// ============================================================================

/// Kinds of pass errors that can occur.
///
/// This enum categorizes the different types of pass operation failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassErrorKind {
    /// A pass is already active and cannot begin a new one.
    PassAlreadyActive,
    /// No pass is active, cannot end or perform pass operations.
    NoActivePass,
    /// The active pass is not the expected type.
    WrongPassType,
    /// The encoder has already been finished.
    EncoderFinished,
}

impl std::fmt::Display for PassErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassErrorKind::PassAlreadyActive => {
                write!(f, "cannot begin a new pass while another pass is active")
            }
            PassErrorKind::NoActivePass => {
                write!(f, "no active pass to end")
            }
            PassErrorKind::WrongPassType => {
                write!(f, "active pass is not the expected type")
            }
            PassErrorKind::EncoderFinished => {
                write!(f, "encoder has already been finished")
            }
        }
    }
}

// ============================================================================
// EncoderStateError
// ============================================================================

/// Error returned when an invalid state transition is attempted.
///
/// This error indicates a programming error - typically trying to:
/// - Begin a pass while another pass is active
/// - Finish the encoder while a pass is active
/// - Use the encoder after it has been finished
///
/// # Example
///
/// ```no_run
/// use renderer_backend::command_encoder::{EncoderState, EncoderStateError};
///
/// let error = EncoderStateError {
///     current: EncoderState::InRenderPass,
///     attempted: EncoderState::InComputePass,
/// };
///
/// assert!(error.to_string().contains("InRenderPass"));
/// assert!(error.to_string().contains("InComputePass"));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderStateError {
    /// The current state of the encoder when the transition was attempted.
    pub current: EncoderState,

    /// The state that was attempted to transition to.
    pub attempted: EncoderState,
}

impl std::fmt::Display for EncoderStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invalid encoder state transition: cannot transition from {} to {}",
            self.current, self.attempted
        )
    }
}

impl std::error::Error for EncoderStateError {}

// ============================================================================
// CommandEncoderDescriptor
// ============================================================================

/// Descriptor for creating a [`TrinityCommandEncoder`].
///
/// This provides a builder-style API for configuring encoder creation options.
/// Currently supports setting an optional label for debugging.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::command_encoder::CommandEncoderDescriptor;
///
/// // Create with default options
/// let desc = CommandEncoderDescriptor::new();
///
/// // Or with a label
/// let desc = CommandEncoderDescriptor::new()
///     .label("Shadow Pass Encoder");
///
/// assert_eq!(desc.label, Some("Shadow Pass Encoder".to_string()));
/// ```
#[derive(Debug, Clone, Default)]
pub struct CommandEncoderDescriptor {
    /// Optional label for debugging.
    ///
    /// This label appears in GPU debugging tools like RenderDoc, PIX, and
    /// wgpu's internal validation messages.
    pub label: Option<String>,
}

impl CommandEncoderDescriptor {
    /// Create a new descriptor with default options.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::command_encoder::CommandEncoderDescriptor;
    ///
    /// let desc = CommandEncoderDescriptor::new();
    /// assert!(desc.label.is_none());
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the label for the command encoder.
    ///
    /// This label is passed to wgpu and appears in GPU debugging tools.
    ///
    /// # Arguments
    ///
    /// * `label` - The label string (anything that implements `Into<String>`)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::command_encoder::CommandEncoderDescriptor;
    ///
    /// let desc = CommandEncoderDescriptor::new()
    ///     .label("GBuffer Pass");
    /// ```
    #[inline]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the label using an Option.
    ///
    /// Useful when the label may or may not be present.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional label string
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renderer_backend::command_encoder::CommandEncoderDescriptor;
    ///
    /// let maybe_label: Option<&str> = Some("Debug Encoder");
    /// let desc = CommandEncoderDescriptor::new()
    ///     .label_opt(maybe_label.map(String::from));
    /// ```
    #[inline]
    pub fn label_opt(mut self, label: Option<String>) -> Self {
        self.label = label;
        self
    }
}

// ============================================================================
// TrinityCommandEncoder
// ============================================================================

/// TRINITY's command encoder wrapper with frame tracking and device reference.
///
/// This struct wraps a `wgpu::CommandEncoder` and adds metadata useful for
/// multi-buffered rendering and debugging:
///
/// - **Frame number**: Tracks which frame this encoder is recording for,
///   enabling frame-based synchronization and profiling.
///
/// - **Device reference**: Keeps an `Arc<wgpu::Device>` so that additional
///   resources (like query sets) can be created during recording.
///
/// - **Label**: Optional debugging label that appears in GPU profiling tools.
///
/// - **State tracking**: Tracks the encoder's lifecycle state for validation.
///
/// # Lifecycle
///
/// A command encoder has four states managed by [`EncoderState`]:
///
/// 1. **Created** - Commands can be recorded (render passes, compute passes, copies)
/// 2. **InRenderPass** - A render pass is active (encoder is borrowed)
/// 3. **InComputePass** - A compute pass is active (encoder is borrowed)
/// 4. **Finished** - The encoder has been consumed to produce a `CommandBuffer`
///
/// ```text
/// TrinityCommandEncoder::new()
///         |
///         v
///     [Created] <────────────────────────────┐
///         |                                  │
///         ├──begin_render_pass──> [InRenderPass] ──end_render_pass──┤
///         │                                  │
///         ├──begin_compute_pass─> [InComputePass] ─end_compute_pass─┘
///         │
///         v
///     finish() -> CommandBuffer -> [Finished]
/// ```
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor, EncoderState};
///
/// # fn example(device: Arc<wgpu::Device>, queue: &wgpu::Queue) {
/// let desc = CommandEncoderDescriptor::new().label("Frame 100 Encoder");
/// let mut encoder = TrinityCommandEncoder::new(&device, &desc, 100);
///
/// assert_eq!(encoder.state(), EncoderState::Created);
/// assert!(encoder.is_recording());
///
/// // Record commands using the inner encoder
/// // encoder.inner_mut().begin_render_pass(...);
///
/// // Finish and submit
/// let command_buffer = encoder.finish();
/// queue.submit(std::iter::once(command_buffer));
/// # }
/// ```
///
/// # Thread Safety
///
/// The encoder itself is `Send + Sync`, but typically command recording
/// happens on a single thread. For multi-threaded command recording,
/// use multiple encoders (one per thread) and submit all command buffers
/// together. State tracking uses atomic operations for thread safety.
pub struct TrinityCommandEncoder {
    /// The underlying wgpu command encoder.
    inner: wgpu::CommandEncoder,

    /// Optional label for debugging.
    label: Option<String>,

    /// Frame number this encoder is associated with.
    ///
    /// Used for frame-based synchronization and profiling correlation.
    frame_number: u64,

    /// Reference to the device that created this encoder.
    ///
    /// Kept for creating additional resources during command recording
    /// (e.g., timestamp query sets, staging buffers).
    device: Arc<wgpu::Device>,

    /// Current state of the encoder lifecycle.
    ///
    /// Uses AtomicU8 for thread-safe state tracking. While the encoder
    /// should typically be used from a single thread, this ensures
    /// state queries are always safe.
    state: AtomicU8,
}

// Explicitly verify Send + Sync bounds
static_assertions::assert_impl_all!(TrinityCommandEncoder: Send, Sync);

impl TrinityCommandEncoder {
    /// Create a new command encoder.
    ///
    /// Creates a `wgpu::CommandEncoder` from the given device and wraps it
    /// with the provided metadata.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the encoder from
    /// * `desc` - Descriptor containing label and other options
    /// * `frame` - Frame number this encoder is recording for
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let desc = CommandEncoderDescriptor::new().label("Main Encoder");
    /// let encoder = TrinityCommandEncoder::new(&device, &desc, 0);
    ///
    /// assert_eq!(encoder.frame_number(), 0);
    /// assert_eq!(encoder.label(), Some("Main Encoder"));
    /// # }
    /// ```
    pub fn new(device: &Arc<wgpu::Device>, desc: &CommandEncoderDescriptor, frame: u64) -> Self {
        let wgpu_desc = wgpu::CommandEncoderDescriptor {
            label: desc.label.as_deref(),
        };

        let inner = device.create_command_encoder(&wgpu_desc);

        Self {
            inner,
            label: desc.label.clone(),
            frame_number: frame,
            device: Arc::clone(device),
            state: AtomicU8::new(EncoderState::Created as u8),
        }
    }

    /// Create a command encoder with a simple label.
    ///
    /// Convenience constructor when you just want to set a label without
    /// creating a descriptor.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the encoder from
    /// * `label` - Optional label for debugging
    /// * `frame` - Frame number this encoder is recording for
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::TrinityCommandEncoder;
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::with_label(&device, Some("Quick Encoder"), 1);
    /// # }
    /// ```
    pub fn with_label(device: &Arc<wgpu::Device>, label: Option<&str>, frame: u64) -> Self {
        let wgpu_desc = wgpu::CommandEncoderDescriptor { label };

        let inner = device.create_command_encoder(&wgpu_desc);

        Self {
            inner,
            label: label.map(String::from),
            frame_number: frame,
            device: Arc::clone(device),
            state: AtomicU8::new(EncoderState::Created as u8),
        }
    }

    /// Get the label of this encoder.
    ///
    /// Returns the label that was set during creation, or `None` if no
    /// label was provided.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let desc = CommandEncoderDescriptor::new().label("My Encoder");
    /// let encoder = TrinityCommandEncoder::new(&device, &desc, 0);
    ///
    /// assert_eq!(encoder.label(), Some("My Encoder"));
    /// # }
    /// ```
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get the frame number this encoder is associated with.
    ///
    /// This is the frame number passed during creation and is useful for:
    /// - Frame-based synchronization with fences
    /// - Correlating GPU profiling data with frame numbers
    /// - Multi-buffered resource management
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     42,
    /// );
    ///
    /// assert_eq!(encoder.frame_number(), 42);
    /// # }
    /// ```
    #[inline]
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Get a reference to the underlying wgpu command encoder.
    ///
    /// Use this for read-only access to the encoder. For recording commands,
    /// use [`inner_mut`](Self::inner_mut) instead.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// let _inner: &wgpu::CommandEncoder = encoder.inner();
    /// # }
    /// ```
    #[inline]
    pub fn inner(&self) -> &wgpu::CommandEncoder {
        &self.inner
    }

    /// Get a mutable reference to the underlying wgpu command encoder.
    ///
    /// Use this to record commands (render passes, compute passes, copies, etc.).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// // Record a compute pass
    /// {
    ///     let mut pass = encoder.inner_mut().begin_compute_pass(
    ///         &wgpu::ComputePassDescriptor {
    ///             label: Some("My Compute Pass"),
    ///             timestamp_writes: None,
    ///         }
    ///     );
    ///     // ... dispatch workgroups ...
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn inner_mut(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.inner
    }

    /// Get a reference to the device that created this encoder.
    ///
    /// Useful for creating additional resources during command recording,
    /// such as timestamp query sets or staging buffers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// // Create a buffer using the encoder's device reference
    /// let buffer = encoder.device().create_buffer(&wgpu::BufferDescriptor {
    ///     label: Some("Staging Buffer"),
    ///     size: 1024,
    ///     usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
    ///     mapped_at_creation: false,
    /// });
    /// # }
    /// ```
    #[inline]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    // =========================================================================
    // State Query Methods
    // =========================================================================

    /// Get the current state of the encoder.
    ///
    /// Returns the current lifecycle state, which can be:
    /// - `Created`: Ready to record commands or begin passes
    /// - `InRenderPass`: A render pass is active
    /// - `InComputePass`: A compute pass is active
    /// - `Finished`: The encoder has been consumed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor, EncoderState};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// assert_eq!(encoder.state(), EncoderState::Created);
    /// # }
    /// ```
    #[inline]
    pub fn state(&self) -> EncoderState {
        EncoderState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Check if the encoder is in the `Created` state (ready to record).
    ///
    /// Returns `true` if the encoder is in its initial state and can:
    /// - Begin render or compute passes
    /// - Record copy/clear commands
    /// - Be finished to produce a command buffer
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// assert!(encoder.is_recording());
    /// # }
    /// ```
    #[inline]
    pub fn is_recording(&self) -> bool {
        self.state() == EncoderState::Created
    }

    /// Check if the encoder is currently in a render or compute pass.
    ///
    /// Returns `true` if a pass is active (either render or compute).
    /// While in a pass, the encoder cannot:
    /// - Begin another pass
    /// - Be finished
    /// - (The pass object borrows the encoder)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// // Initially not in a pass
    /// assert!(!encoder.is_in_pass());
    /// # }
    /// ```
    #[inline]
    pub fn is_in_pass(&self) -> bool {
        self.state().is_in_pass()
    }

    /// Check if the encoder has been finished (consumed).
    ///
    /// Returns `true` if `finish()` has been called and the encoder
    /// has produced a command buffer. This is a terminal state.
    ///
    /// # Note
    ///
    /// Since `finish()` consumes `self`, this method is primarily useful
    /// when checking state through a reference before a potential finish,
    /// or in testing scenarios.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// // Not finished yet
    /// assert!(!encoder.is_finished());
    /// # }
    /// ```
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.state() == EncoderState::Finished
    }

    // =========================================================================
    // State Transition Methods (Internal)
    // =========================================================================

    /// Attempt to transition to a new state.
    ///
    /// Validates the transition and returns an error if invalid.
    /// In debug builds, invalid transitions also trigger a debug_assert.
    ///
    /// # Valid Transitions
    ///
    /// - `Created` -> `InRenderPass` (begin_render_pass)
    /// - `Created` -> `InComputePass` (begin_compute_pass)
    /// - `Created` -> `Finished` (finish)
    /// - `InRenderPass` -> `Created` (end_render_pass)
    /// - `InComputePass` -> `Created` (end_compute_pass)
    ///
    /// # Arguments
    ///
    /// * `new_state` - The state to transition to
    ///
    /// # Errors
    ///
    /// Returns `EncoderStateError` if the transition is invalid.
    fn transition_to(&self, new_state: EncoderState) -> Result<(), EncoderStateError> {
        let current = self.state();

        let valid = match (current, new_state) {
            // From Created, can go to any pass or finish
            (EncoderState::Created, EncoderState::InRenderPass) => true,
            (EncoderState::Created, EncoderState::InComputePass) => true,
            (EncoderState::Created, EncoderState::Finished) => true,
            // From a pass, can only go back to Created
            (EncoderState::InRenderPass, EncoderState::Created) => true,
            (EncoderState::InComputePass, EncoderState::Created) => true,
            // All other transitions are invalid
            _ => false,
        };

        if valid {
            self.state.store(new_state as u8, Ordering::Release);
            Ok(())
        } else {
            let error = EncoderStateError {
                current,
                attempted: new_state,
            };
            debug_assert!(
                false,
                "Invalid encoder state transition: {:?} -> {:?}",
                current, new_state
            );
            Err(error)
        }
    }

    /// Mark the start of a render pass.
    ///
    /// Transitions state from `Created` to `InRenderPass`.
    /// This should be called before the wgpu render pass is created.
    ///
    /// # Panics (Debug Only)
    ///
    /// In debug builds, panics if not in `Created` state.
    #[inline]
    pub fn begin_render_pass_internal(&self) {
        let _ = self.transition_to(EncoderState::InRenderPass);
    }

    /// Mark the end of a render pass.
    ///
    /// Transitions state from `InRenderPass` back to `Created`.
    /// This should be called after the wgpu render pass is dropped.
    ///
    /// # Panics (Debug Only)
    ///
    /// In debug builds, panics if not in `InRenderPass` state.
    #[inline]
    pub fn end_render_pass_internal(&self) {
        let _ = self.transition_to(EncoderState::Created);
    }

    /// Mark the start of a compute pass.
    ///
    /// Transitions state from `Created` to `InComputePass`.
    /// This should be called before the wgpu compute pass is created.
    ///
    /// # Panics (Debug Only)
    ///
    /// In debug builds, panics if not in `Created` state.
    #[inline]
    pub fn begin_compute_pass_internal(&self) {
        let _ = self.transition_to(EncoderState::InComputePass);
    }

    /// Mark the end of a compute pass.
    ///
    /// Transitions state from `InComputePass` back to `Created`.
    /// This should be called after the wgpu compute pass is dropped.
    ///
    /// # Panics (Debug Only)
    ///
    /// In debug builds, panics if not in `InComputePass` state.
    #[inline]
    pub fn end_compute_pass_internal(&self) {
        let _ = self.transition_to(EncoderState::Created);
    }

    // =========================================================================
    // Pass Tracking Methods (T-WGPU-P4.1.3)
    // =========================================================================

    /// Get the type of the currently active pass.
    ///
    /// Returns [`ActivePassType::None`] if no pass is active,
    /// [`ActivePassType::Render`] if a render pass is active, or
    /// [`ActivePassType::Compute`] if a compute pass is active.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor, ActivePassType};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
    ///
    /// assert_eq!(encoder.active_pass(), ActivePassType::None);
    /// # }
    /// ```
    #[inline]
    pub fn active_pass(&self) -> ActivePassType {
        ActivePassType::from_encoder_state(self.state())
    }

    /// Check if there is an active pass (render or compute).
    ///
    /// Returns `true` if either a render pass or compute pass is currently active.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
    ///
    /// assert!(!encoder.has_active_pass());
    /// # }
    /// ```
    #[inline]
    pub fn has_active_pass(&self) -> bool {
        self.active_pass().is_active()
    }

    /// Begin a render pass with validation.
    ///
    /// This method checks that no pass is currently active before beginning a
    /// render pass. If a pass is already active, returns an error instead of
    /// panicking.
    ///
    /// # Arguments
    ///
    /// * `desc` - The render pass descriptor (not used directly, validation only)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the render pass can begin (state is now `InRenderPass`)
    /// - `Err(PassError)` if a pass is already active or encoder is finished
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let mut encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
    ///
    /// // First render pass succeeds
    /// assert!(encoder.begin_render_pass_checked().is_ok());
    ///
    /// // Second render pass fails while first is active
    /// assert!(encoder.begin_render_pass_checked().is_err());
    /// # }
    /// ```
    pub fn begin_render_pass_checked(&self) -> Result<(), PassError> {
        let current = self.state();
        match current {
            EncoderState::Created => {
                self.state.store(EncoderState::InRenderPass as u8, Ordering::Release);
                Ok(())
            }
            EncoderState::InRenderPass | EncoderState::InComputePass => {
                Err(PassError::already_active())
            }
            EncoderState::Finished => {
                Err(PassError::encoder_finished())
            }
        }
    }

    /// Begin a compute pass with validation.
    ///
    /// This method checks that no pass is currently active before beginning a
    /// compute pass. If a pass is already active, returns an error instead of
    /// panicking.
    ///
    /// # Arguments
    ///
    /// * `desc` - The compute pass descriptor (not used directly, validation only)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the compute pass can begin (state is now `InComputePass`)
    /// - `Err(PassError)` if a pass is already active or encoder is finished
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let mut encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
    ///
    /// // Compute pass succeeds
    /// assert!(encoder.begin_compute_pass_checked().is_ok());
    ///
    /// // Another pass fails while compute is active
    /// assert!(encoder.begin_render_pass_checked().is_err());
    /// # }
    /// ```
    pub fn begin_compute_pass_checked(&self) -> Result<(), PassError> {
        let current = self.state();
        match current {
            EncoderState::Created => {
                self.state.store(EncoderState::InComputePass as u8, Ordering::Release);
                Ok(())
            }
            EncoderState::InRenderPass | EncoderState::InComputePass => {
                Err(PassError::already_active())
            }
            EncoderState::Finished => {
                Err(PassError::encoder_finished())
            }
        }
    }

    /// End the current pass.
    ///
    /// This method ends whatever pass is currently active (render or compute).
    /// If no pass is active, returns an error.
    ///
    /// # Returns
    ///
    /// - `Ok(ActivePassType)` with the type of pass that was ended
    /// - `Err(PassError)` if no pass is active
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor, ActivePassType};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let mut encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
    ///
    /// // Begin a render pass
    /// encoder.begin_render_pass_checked().unwrap();
    ///
    /// // End it
    /// let ended = encoder.end_pass().unwrap();
    /// assert_eq!(ended, ActivePassType::Render);
    ///
    /// // Ending again fails
    /// assert!(encoder.end_pass().is_err());
    /// # }
    /// ```
    pub fn end_pass(&self) -> Result<ActivePassType, PassError> {
        let current = self.state();
        match current {
            EncoderState::InRenderPass => {
                self.state.store(EncoderState::Created as u8, Ordering::Release);
                Ok(ActivePassType::Render)
            }
            EncoderState::InComputePass => {
                self.state.store(EncoderState::Created as u8, Ordering::Release);
                Ok(ActivePassType::Compute)
            }
            EncoderState::Created => {
                Err(PassError::no_active())
            }
            EncoderState::Finished => {
                Err(PassError::encoder_finished())
            }
        }
    }

    /// End a specific type of pass with type checking.
    ///
    /// This method ends the current pass only if it matches the expected type.
    /// Returns an error if the wrong type of pass is active.
    ///
    /// # Arguments
    ///
    /// * `expected` - The expected type of pass to end
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the pass was ended successfully
    /// - `Err(PassError::NoActivePass)` if no pass is active
    /// - `Err(PassError::WrongPassType)` if the wrong type of pass is active
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor, ActivePassType};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let mut encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
    ///
    /// // Begin a render pass
    /// encoder.begin_render_pass_checked().unwrap();
    ///
    /// // Ending as compute fails
    /// assert!(encoder.end_pass_typed(ActivePassType::Compute).is_err());
    ///
    /// // Ending as render succeeds
    /// assert!(encoder.end_pass_typed(ActivePassType::Render).is_ok());
    /// # }
    /// ```
    pub fn end_pass_typed(&self, expected: ActivePassType) -> Result<(), PassError> {
        let current = self.active_pass();

        if current == ActivePassType::None {
            return Err(PassError::no_active());
        }

        if current != expected {
            return Err(PassError::wrong_type());
        }

        self.state.store(EncoderState::Created as u8, Ordering::Release);
        Ok(())
    }

    /// Automatically end any active pass if one exists, with a debug warning.
    ///
    /// This method is called internally by `finish()` to ensure passes are
    /// properly ended. If a pass is active, it ends it and emits a debug warning.
    ///
    /// # Returns
    ///
    /// `true` if a pass was implicitly ended, `false` otherwise.
    ///
    /// # Debug Warning
    ///
    /// When a pass is implicitly ended, a warning is printed via `eprintln!`
    /// to help identify code that doesn't properly end passes.
    fn auto_end_pass_if_active(&self) -> bool {
        let current = self.state();
        match current {
            EncoderState::InRenderPass => {
                eprintln!(
                    "[TRINITY WARNING] Encoder '{}' (frame {}) implicitly ending render pass in finish(). \
                     Please explicitly end passes before calling finish().",
                    self.label.as_deref().unwrap_or("<unnamed>"),
                    self.frame_number
                );
                self.state.store(EncoderState::Created as u8, Ordering::Release);
                true
            }
            EncoderState::InComputePass => {
                eprintln!(
                    "[TRINITY WARNING] Encoder '{}' (frame {}) implicitly ending compute pass in finish(). \
                     Please explicitly end passes before calling finish().",
                    self.label.as_deref().unwrap_or("<unnamed>"),
                    self.frame_number
                );
                self.state.store(EncoderState::Created as u8, Ordering::Release);
                true
            }
            _ => false,
        }
    }

    /// Finish recording and produce a command buffer.
    ///
    /// Consumes the encoder and returns a `wgpu::CommandBuffer` ready for
    /// submission to a queue. Transitions state to `Finished`.
    ///
    /// If a pass is still active when `finish()` is called, it will be
    /// implicitly ended and a debug warning will be printed. This allows
    /// the encoder to finish successfully but alerts developers to fix
    /// their code.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>, queue: &wgpu::Queue) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new().label("Frame Encoder"),
    ///     0,
    /// );
    ///
    /// // ... record commands ...
    ///
    /// let command_buffer = encoder.finish();
    /// queue.submit(std::iter::once(command_buffer));
    /// # }
    /// ```
    #[inline]
    pub fn finish(self) -> wgpu::CommandBuffer {
        // Auto-end any active pass with a warning
        self.auto_end_pass_if_active();

        // Validate state transition (debug builds will panic on invalid)
        let _ = self.transition_to(EncoderState::Finished);
        self.inner.finish()
    }

    /// Finish recording and produce a command buffer, requiring no active pass.
    ///
    /// Unlike `finish()`, this method returns an error if a pass is still
    /// active instead of implicitly ending it.
    ///
    /// # Returns
    ///
    /// - `Ok(CommandBuffer)` if successful
    /// - `Err(PassError::PassAlreadyActive)` if a pass is still active
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>, queue: &wgpu::Queue) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// // This will error if we forgot to end a pass
    /// let command_buffer = encoder.finish_checked().expect("Pass still active!");
    /// queue.submit(std::iter::once(command_buffer));
    /// # }
    /// ```
    pub fn finish_checked(self) -> Result<wgpu::CommandBuffer, PassError> {
        let current = self.state();
        match current {
            EncoderState::Created => {
                self.state.store(EncoderState::Finished as u8, Ordering::Release);
                Ok(self.inner.finish())
            }
            EncoderState::InRenderPass | EncoderState::InComputePass => {
                Err(PassError::already_active())
            }
            EncoderState::Finished => {
                Err(PassError::encoder_finished())
            }
        }
    }

    // =========================================================================
    // Command Buffer Finalization Methods (T-WGPU-P4.1.4)
    // =========================================================================

    /// Consume the encoder and return the underlying command buffer.
    ///
    /// This method explicitly consumes the `TrinityCommandEncoder` (takes ownership)
    /// and returns the `wgpu::CommandBuffer` ready for submission to a queue.
    ///
    /// Unlike `finish()`, this method:
    /// - Returns a `Result` for explicit error handling
    /// - Validates that the encoder is not already finished
    /// - Validates that no pass is currently active
    ///
    /// # Returns
    ///
    /// - `Ok(wgpu::CommandBuffer)` - The finished command buffer
    /// - `Err(EncoderStateError)` - If the encoder is in an invalid state
    ///
    /// # Errors
    ///
    /// Returns `EncoderStateError` if:
    /// - A render or compute pass is still active
    /// - The encoder has already been finished
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>, queue: &wgpu::Queue) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new().label("Frame Encoder"),
    ///     0,
    /// );
    ///
    /// // Record commands...
    ///
    /// // Explicitly consume and get the command buffer
    /// let command_buffer = encoder.into_command_buffer()
    ///     .expect("Failed to finalize command buffer");
    ///
    /// queue.submit(std::iter::once(command_buffer));
    /// # }
    /// ```
    pub fn into_command_buffer(self) -> Result<wgpu::CommandBuffer, EncoderStateError> {
        let current = self.state();
        match current {
            EncoderState::Created => {
                self.state.store(EncoderState::Finished as u8, Ordering::Release);
                Ok(self.inner.finish())
            }
            EncoderState::InRenderPass => {
                Err(EncoderStateError {
                    current: EncoderState::InRenderPass,
                    attempted: EncoderState::Finished,
                })
            }
            EncoderState::InComputePass => {
                Err(EncoderStateError {
                    current: EncoderState::InComputePass,
                    attempted: EncoderState::Finished,
                })
            }
            EncoderState::Finished => {
                Err(EncoderStateError {
                    current: EncoderState::Finished,
                    attempted: EncoderState::Finished,
                })
            }
        }
    }

    /// Submit command buffers to a queue.
    ///
    /// This is a static helper method that submits one or more command buffers
    /// to the given queue. It wraps `wgpu::Queue::submit()` for convenience.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue to submit to
    /// * `buffers` - Iterator of command buffers to submit
    ///
    /// # Returns
    ///
    /// The `wgpu::SubmissionIndex` from the queue submission, which can be used
    /// to track when the GPU has finished executing the commands.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>, queue: &wgpu::Queue) {
    /// let encoder1 = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
    /// let encoder2 = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
    ///
    /// let cb1 = encoder1.finish();
    /// let cb2 = encoder2.finish();
    ///
    /// // Submit multiple command buffers at once
    /// let index = TrinityCommandEncoder::submit_to_queue(queue, [cb1, cb2]);
    /// # }
    /// ```
    #[inline]
    pub fn submit_to_queue<I>(queue: &wgpu::Queue, buffers: I) -> wgpu::SubmissionIndex
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        queue.submit(buffers)
    }

    /// Finish recording and submit the command buffer to a queue in one step.
    ///
    /// This convenience method combines `into_command_buffer()` and queue submission
    /// into a single call. It consumes the encoder and submits the resulting
    /// command buffer to the provided queue.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue to submit to
    ///
    /// # Returns
    ///
    /// - `Ok(wgpu::SubmissionIndex)` - The submission index for tracking GPU completion
    /// - `Err(EncoderStateError)` - If the encoder is in an invalid state
    ///
    /// # Errors
    ///
    /// Returns `EncoderStateError` if:
    /// - A render or compute pass is still active
    /// - The encoder has already been finished
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>, queue: &wgpu::Queue) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new().label("Quick Submit"),
    ///     0,
    /// );
    ///
    /// // Record commands...
    ///
    /// // Finish and submit in one call
    /// let submission_index = encoder.finish_and_submit(queue)
    ///     .expect("Failed to submit commands");
    ///
    /// // Optionally wait for GPU completion
    /// // device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission_index));
    /// # }
    /// ```
    pub fn finish_and_submit(self, queue: &wgpu::Queue) -> Result<wgpu::SubmissionIndex, EncoderStateError> {
        let command_buffer = self.into_command_buffer()?;
        Ok(queue.submit(std::iter::once(command_buffer)))
    }

    /// Finish recording with auto-ending of passes and submit to a queue.
    ///
    /// This is a more lenient version of `finish_and_submit()` that will
    /// automatically end any active pass (with a warning) before submitting.
    /// Use this when you want to ensure submission succeeds even if a pass
    /// wasn't properly ended.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue to submit to
    ///
    /// # Returns
    ///
    /// The `wgpu::SubmissionIndex` from the queue submission.
    ///
    /// # Panics
    ///
    /// May panic in debug builds if the encoder is in the `Finished` state.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>, queue: &wgpu::Queue) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new().label("Lenient Submit"),
    ///     0,
    /// );
    ///
    /// // Record commands... maybe forget to end a pass
    ///
    /// // Submit anyway (with warning if pass was active)
    /// let submission_index = encoder.finish_and_submit_lenient(queue);
    /// # }
    /// ```
    #[inline]
    pub fn finish_and_submit_lenient(self, queue: &wgpu::Queue) -> wgpu::SubmissionIndex {
        let command_buffer = self.finish();
        queue.submit(std::iter::once(command_buffer))
    }

    /// Insert a debug marker into the command stream.
    ///
    /// Debug markers appear in GPU debugging tools (RenderDoc, PIX, etc.)
    /// and are useful for identifying specific points in command execution.
    ///
    /// # Arguments
    ///
    /// * `label` - The marker label
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// encoder.insert_debug_marker("Before Shadow Pass");
    /// // ... shadow pass ...
    /// encoder.insert_debug_marker("After Shadow Pass");
    /// # }
    /// ```
    #[inline]
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.inner.insert_debug_marker(label);
    }

    /// Push a debug group onto the stack.
    ///
    /// Debug groups appear as hierarchical regions in GPU debugging tools.
    /// Must be paired with a matching [`pop_debug_group`](Self::pop_debug_group) call.
    ///
    /// # Arguments
    ///
    /// * `label` - The group label
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// encoder.push_debug_group("GBuffer Pass");
    /// // ... gbuffer commands ...
    /// encoder.pop_debug_group();
    /// # }
    /// ```
    #[inline]
    pub fn push_debug_group(&mut self, label: &str) {
        self.inner.push_debug_group(label);
    }

    /// Pop a debug group from the stack.
    ///
    /// Must be paired with a previous [`push_debug_group`](Self::push_debug_group) call.
    ///
    /// # Example
    ///
    /// See [`push_debug_group`](Self::push_debug_group) for a complete example.
    #[inline]
    pub fn pop_debug_group(&mut self) {
        self.inner.pop_debug_group();
    }

    /// Copy data from one buffer to another.
    ///
    /// # Arguments
    ///
    /// * `source` - Source buffer
    /// * `source_offset` - Offset in bytes from the start of the source buffer
    /// * `destination` - Destination buffer
    /// * `destination_offset` - Offset in bytes from the start of the destination buffer
    /// * `copy_size` - Number of bytes to copy
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `source_offset + copy_size` exceeds the source buffer size
    /// - `destination_offset + copy_size` exceeds the destination buffer size
    /// - Offsets or sizes are not aligned to `COPY_BUFFER_ALIGNMENT` (4 bytes)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>, src: &wgpu::Buffer, dst: &wgpu::Buffer) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// encoder.copy_buffer_to_buffer(src, 0, dst, 0, 1024);
    /// # }
    /// ```
    #[inline]
    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &wgpu::Buffer,
        source_offset: wgpu::BufferAddress,
        destination: &wgpu::Buffer,
        destination_offset: wgpu::BufferAddress,
        copy_size: wgpu::BufferAddress,
    ) {
        self.inner.copy_buffer_to_buffer(
            source,
            source_offset,
            destination,
            destination_offset,
            copy_size,
        );
    }

    /// Clear a buffer to zero.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to clear
    /// * `offset` - Offset in bytes from the start of the buffer
    /// * `size` - Number of bytes to clear, or `None` to clear from offset to end
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `offset + size` exceeds the buffer size
    /// - Offset or size is not aligned to `COPY_BUFFER_ALIGNMENT` (4 bytes)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use renderer_backend::command_encoder::{TrinityCommandEncoder, CommandEncoderDescriptor};
    ///
    /// # fn example(device: Arc<wgpu::Device>, buffer: &wgpu::Buffer) {
    /// let mut encoder = TrinityCommandEncoder::new(
    ///     &device,
    ///     &CommandEncoderDescriptor::new(),
    ///     0,
    /// );
    ///
    /// // Clear first 256 bytes
    /// encoder.clear_buffer(buffer, 0, Some(256));
    ///
    /// // Clear entire buffer from offset 256
    /// encoder.clear_buffer(buffer, 256, None);
    /// # }
    /// ```
    #[inline]
    pub fn clear_buffer(
        &mut self,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferAddress>,
    ) {
        self.inner.clear_buffer(buffer, offset, size);
    }
}

impl std::fmt::Debug for TrinityCommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrinityCommandEncoder")
            .field("label", &self.label)
            .field("frame_number", &self.frame_number)
            .field("state", &self.state())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for TrinityCommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.label {
            Some(label) => write!(f, "TrinityCommandEncoder('{}', frame {})", label, self.frame_number),
            None => write!(f, "TrinityCommandEncoder(frame {})", self.frame_number),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test device.
    /// Returns None if no GPU adapter is available (CI/headless).
    fn test_device() -> Option<Arc<wgpu::Device>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, _queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        Some(Arc::new(device))
    }

    // -------------------------------------------------------------------------
    // CommandEncoderDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_new() {
        let desc = CommandEncoderDescriptor::new();
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_descriptor_default() {
        let desc = CommandEncoderDescriptor::default();
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_descriptor_label() {
        let desc = CommandEncoderDescriptor::new().label("Test Label");
        assert_eq!(desc.label, Some("Test Label".to_string()));
    }

    #[test]
    fn test_descriptor_label_string() {
        let desc = CommandEncoderDescriptor::new().label(String::from("String Label"));
        assert_eq!(desc.label, Some("String Label".to_string()));
    }

    #[test]
    fn test_descriptor_label_opt_some() {
        let desc = CommandEncoderDescriptor::new().label_opt(Some("Optional Label".to_string()));
        assert_eq!(desc.label, Some("Optional Label".to_string()));
    }

    #[test]
    fn test_descriptor_label_opt_none() {
        let desc = CommandEncoderDescriptor::new().label_opt(None);
        assert!(desc.label.is_none());
    }

    #[test]
    fn test_descriptor_clone() {
        let desc = CommandEncoderDescriptor::new().label("Clone Me");
        let cloned = desc.clone();
        assert_eq!(cloned.label, Some("Clone Me".to_string()));
    }

    #[test]
    fn test_descriptor_debug() {
        let desc = CommandEncoderDescriptor::new().label("Debug");
        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("Debug"));
    }

    // -------------------------------------------------------------------------
    // TrinityCommandEncoder tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_new() {
        let Some(device) = test_device() else {
            return;
        };

        let desc = CommandEncoderDescriptor::new().label("New Encoder");
        let encoder = TrinityCommandEncoder::new(&device, &desc, 42);

        assert_eq!(encoder.label(), Some("New Encoder"));
        assert_eq!(encoder.frame_number(), 42);
    }

    #[test]
    fn test_encoder_with_label() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::with_label(&device, Some("Quick Label"), 100);
        assert_eq!(encoder.label(), Some("Quick Label"));
        assert_eq!(encoder.frame_number(), 100);
    }

    #[test]
    fn test_encoder_with_label_none() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::with_label(&device, None, 0);
        assert!(encoder.label().is_none());
        assert_eq!(encoder.frame_number(), 0);
    }

    #[test]
    fn test_encoder_inner_accessors() {
        let Some(device) = test_device() else {
            return;
        };

        let mut encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);

        // Should be able to get shared reference
        let _inner: &wgpu::CommandEncoder = encoder.inner();

        // Should be able to get mutable reference
        let _inner_mut: &mut wgpu::CommandEncoder = encoder.inner_mut();
    }

    #[test]
    fn test_encoder_device_reference() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);

        // Device reference should be valid
        let _dev: &Arc<wgpu::Device> = encoder.device();

        // Should be able to create resources with it
        let _buffer = encoder.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Buffer"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
    }

    #[test]
    fn test_encoder_finish() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);

        // Finish should produce a valid command buffer
        let _command_buffer: wgpu::CommandBuffer = encoder.finish();
    }

    #[test]
    fn test_encoder_debug_markers() {
        let Some(device) = test_device() else {
            return;
        };

        let mut encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);

        // Debug markers should not panic
        encoder.push_debug_group("Outer Group");
        encoder.push_debug_group("Inner Group");
        encoder.insert_debug_marker("Marker");
        encoder.pop_debug_group();
        encoder.pop_debug_group();

        let _cb = encoder.finish();
    }

    #[test]
    fn test_encoder_copy_buffer_to_buffer() {
        let Some(device) = test_device() else {
            return;
        };

        let src = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Source"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destination"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);

        encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, 256);

        let _cb = encoder.finish();
    }

    #[test]
    fn test_encoder_clear_buffer() {
        let Some(device) = test_device() else {
            return;
        };

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Clear Target"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);

        // Clear with explicit size
        encoder.clear_buffer(&buffer, 0, Some(128));

        // Clear to end
        encoder.clear_buffer(&buffer, 128, None);

        let _cb = encoder.finish();
    }

    #[test]
    fn test_encoder_debug_fmt() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Debug Test"),
            42,
        );

        let debug_str = format!("{:?}", encoder);
        assert!(debug_str.contains("TrinityCommandEncoder"));
        assert!(debug_str.contains("Debug Test"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_encoder_display_with_label() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Display Test"),
            99,
        );

        let display_str = format!("{}", encoder);
        assert_eq!(display_str, "TrinityCommandEncoder('Display Test', frame 99)");
    }

    #[test]
    fn test_encoder_display_without_label() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 7);

        let display_str = format!("{}", encoder);
        assert_eq!(display_str, "TrinityCommandEncoder(frame 7)");
    }

    #[test]
    fn test_encoder_frame_number_tracking() {
        let Some(device) = test_device() else {
            return;
        };

        // Test various frame numbers
        for frame in [0, 1, 100, u64::MAX / 2, u64::MAX] {
            let encoder = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), frame);
            assert_eq!(encoder.frame_number(), frame);
        }
    }

    #[test]
    fn test_encoder_multiple_instances() {
        let Some(device) = test_device() else {
            return;
        };

        // Create multiple encoders (simulating multi-frame buffering)
        let encoder1 = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Frame 0"),
            0,
        );
        let encoder2 = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Frame 1"),
            1,
        );
        let encoder3 = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Frame 2"),
            2,
        );

        assert_eq!(encoder1.frame_number(), 0);
        assert_eq!(encoder2.frame_number(), 1);
        assert_eq!(encoder3.frame_number(), 2);

        // All should finish successfully
        let _cb1 = encoder1.finish();
        let _cb2 = encoder2.finish();
        let _cb3 = encoder3.finish();
    }

    // -------------------------------------------------------------------------
    // Integration tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_compute_pass_integration() {
        let Some(device) = test_device() else {
            return;
        };

        let mut encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Compute Integration"),
            0,
        );

        {
            let _pass = encoder.inner_mut().begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Test Compute Pass"),
                timestamp_writes: None,
            });
            // Pass is dropped here, ending it
        }

        let _cb = encoder.finish();
    }

    #[test]
    fn test_encoder_render_pass_integration() {
        let Some(device) = test_device() else {
            return;
        };

        // Create a render target
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Target"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Render Integration"),
            0,
        );

        {
            let _pass = encoder.inner_mut().begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Test Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            // Pass is dropped here, ending it
        }

        let _cb = encoder.finish();
    }

    // -------------------------------------------------------------------------
    // EncoderState tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_state_from_u8() {
        assert_eq!(EncoderState::from_u8(0), EncoderState::Created);
        assert_eq!(EncoderState::from_u8(1), EncoderState::InRenderPass);
        assert_eq!(EncoderState::from_u8(2), EncoderState::InComputePass);
        assert_eq!(EncoderState::from_u8(3), EncoderState::Finished);
        // Invalid values should default to Created
        assert_eq!(EncoderState::from_u8(255), EncoderState::Created);
    }

    #[test]
    fn test_encoder_state_can_begin_pass() {
        assert!(EncoderState::Created.can_begin_pass());
        assert!(!EncoderState::InRenderPass.can_begin_pass());
        assert!(!EncoderState::InComputePass.can_begin_pass());
        assert!(!EncoderState::Finished.can_begin_pass());
    }

    #[test]
    fn test_encoder_state_can_finish() {
        assert!(EncoderState::Created.can_finish());
        assert!(!EncoderState::InRenderPass.can_finish());
        assert!(!EncoderState::InComputePass.can_finish());
        assert!(!EncoderState::Finished.can_finish());
    }

    #[test]
    fn test_encoder_state_is_in_pass() {
        assert!(!EncoderState::Created.is_in_pass());
        assert!(EncoderState::InRenderPass.is_in_pass());
        assert!(EncoderState::InComputePass.is_in_pass());
        assert!(!EncoderState::Finished.is_in_pass());
    }

    #[test]
    fn test_encoder_state_display() {
        assert_eq!(EncoderState::Created.to_string(), "Created");
        assert_eq!(EncoderState::InRenderPass.to_string(), "InRenderPass");
        assert_eq!(EncoderState::InComputePass.to_string(), "InComputePass");
        assert_eq!(EncoderState::Finished.to_string(), "Finished");
    }

    #[test]
    fn test_encoder_state_debug() {
        let debug_str = format!("{:?}", EncoderState::Created);
        assert!(debug_str.contains("Created"));
    }

    #[test]
    fn test_encoder_state_clone_copy() {
        let state = EncoderState::InRenderPass;
        let cloned = state.clone();
        let copied = state;
        assert_eq!(state, cloned);
        assert_eq!(state, copied);
    }

    #[test]
    fn test_encoder_state_eq_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(EncoderState::Created);
        set.insert(EncoderState::InRenderPass);
        set.insert(EncoderState::InComputePass);
        set.insert(EncoderState::Finished);

        assert_eq!(set.len(), 4);
        assert!(set.contains(&EncoderState::Created));
        assert!(set.contains(&EncoderState::InRenderPass));
        assert!(set.contains(&EncoderState::InComputePass));
        assert!(set.contains(&EncoderState::Finished));
    }

    // -------------------------------------------------------------------------
    // EncoderStateError tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_state_error_display() {
        let error = EncoderStateError {
            current: EncoderState::InRenderPass,
            attempted: EncoderState::InComputePass,
        };

        let display = error.to_string();
        assert!(display.contains("InRenderPass"));
        assert!(display.contains("InComputePass"));
        assert!(display.contains("Invalid encoder state transition"));
    }

    #[test]
    fn test_encoder_state_error_debug() {
        let error = EncoderStateError {
            current: EncoderState::Created,
            attempted: EncoderState::Finished,
        };

        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("EncoderStateError"));
        assert!(debug_str.contains("Created"));
        assert!(debug_str.contains("Finished"));
    }

    #[test]
    fn test_encoder_state_error_is_error() {
        let error = EncoderStateError {
            current: EncoderState::Finished,
            attempted: EncoderState::Created,
        };

        // Test std::error::Error implementation
        let _: &dyn std::error::Error = &error;
    }

    // -------------------------------------------------------------------------
    // TrinityCommandEncoder state tracking tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_initial_state_is_created() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        assert_eq!(encoder.state(), EncoderState::Created);
        assert!(encoder.is_recording());
        assert!(!encoder.is_in_pass());
        assert!(!encoder.is_finished());
    }

    #[test]
    fn test_encoder_render_pass_state_transitions() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Start in Created state
        assert_eq!(encoder.state(), EncoderState::Created);
        assert!(encoder.is_recording());

        // Transition to InRenderPass
        encoder.begin_render_pass_internal();
        assert_eq!(encoder.state(), EncoderState::InRenderPass);
        assert!(!encoder.is_recording());
        assert!(encoder.is_in_pass());

        // Transition back to Created
        encoder.end_render_pass_internal();
        assert_eq!(encoder.state(), EncoderState::Created);
        assert!(encoder.is_recording());
        assert!(!encoder.is_in_pass());

        // Finish transitions to Finished state
        let _cb = encoder.finish();
        // (encoder is consumed, can't check state after finish)
    }

    #[test]
    fn test_encoder_compute_pass_state_transitions() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Start in Created state
        assert_eq!(encoder.state(), EncoderState::Created);

        // Transition to InComputePass
        encoder.begin_compute_pass_internal();
        assert_eq!(encoder.state(), EncoderState::InComputePass);
        assert!(encoder.is_in_pass());

        // Transition back to Created
        encoder.end_compute_pass_internal();
        assert_eq!(encoder.state(), EncoderState::Created);
        assert!(!encoder.is_in_pass());
    }

    #[test]
    fn test_encoder_multiple_pass_cycles() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // First render pass cycle
        encoder.begin_render_pass_internal();
        assert_eq!(encoder.state(), EncoderState::InRenderPass);
        encoder.end_render_pass_internal();
        assert_eq!(encoder.state(), EncoderState::Created);

        // Compute pass cycle
        encoder.begin_compute_pass_internal();
        assert_eq!(encoder.state(), EncoderState::InComputePass);
        encoder.end_compute_pass_internal();
        assert_eq!(encoder.state(), EncoderState::Created);

        // Another render pass cycle
        encoder.begin_render_pass_internal();
        encoder.end_render_pass_internal();
        assert_eq!(encoder.state(), EncoderState::Created);

        // Finish
        let _cb = encoder.finish();
    }

    #[test]
    fn test_encoder_transition_validation_created_to_render() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Valid transition: Created -> InRenderPass
        let result = encoder.transition_to(EncoderState::InRenderPass);
        assert!(result.is_ok());
        assert_eq!(encoder.state(), EncoderState::InRenderPass);
    }

    #[test]
    fn test_encoder_transition_validation_created_to_compute() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Valid transition: Created -> InComputePass
        let result = encoder.transition_to(EncoderState::InComputePass);
        assert!(result.is_ok());
        assert_eq!(encoder.state(), EncoderState::InComputePass);
    }

    #[test]
    fn test_encoder_transition_validation_render_to_created() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_render_pass_internal();

        // Valid transition: InRenderPass -> Created
        let result = encoder.transition_to(EncoderState::Created);
        assert!(result.is_ok());
        assert_eq!(encoder.state(), EncoderState::Created);
    }

    #[test]
    fn test_encoder_transition_validation_compute_to_created() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_compute_pass_internal();

        // Valid transition: InComputePass -> Created
        let result = encoder.transition_to(EncoderState::Created);
        assert!(result.is_ok());
        assert_eq!(encoder.state(), EncoderState::Created);
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)] // Invalid transitions panic in debug
    fn test_encoder_invalid_transition_render_to_compute() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_render_pass_internal();

        // Invalid transition: InRenderPass -> InComputePass
        let result = encoder.transition_to(EncoderState::InComputePass);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.current, EncoderState::InRenderPass);
        assert_eq!(err.attempted, EncoderState::InComputePass);
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore)] // Invalid transitions panic in debug
    fn test_encoder_invalid_transition_finished_to_created() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Manually set to finished state for testing
        encoder.state.store(EncoderState::Finished as u8, Ordering::Release);

        // Invalid transition: Finished -> Created
        let result = encoder.transition_to(EncoderState::Created);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.current, EncoderState::Finished);
        assert_eq!(err.attempted, EncoderState::Created);
    }

    #[test]
    fn test_encoder_state_in_debug_format() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("State Debug Test"),
            42,
        );

        let debug_str = format!("{:?}", encoder);
        assert!(debug_str.contains("Created")); // State should be in debug output
        assert!(debug_str.contains("State Debug Test"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_encoder_state_thread_safety() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // State reads should be atomic and thread-safe
        // (testing the API, not actual concurrent access)
        for _ in 0..100 {
            let _ = encoder.state();
            let _ = encoder.is_recording();
            let _ = encoder.is_in_pass();
            let _ = encoder.is_finished();
        }

        assert_eq!(encoder.state(), EncoderState::Created);
    }

    // -------------------------------------------------------------------------
    // ActivePassType tests (T-WGPU-P4.1.3)
    // -------------------------------------------------------------------------

    #[test]
    fn test_active_pass_type_none() {
        let pass_type = ActivePassType::None;
        assert!(!pass_type.is_active());
        assert_eq!(pass_type.to_string(), "None");
    }

    #[test]
    fn test_active_pass_type_render() {
        let pass_type = ActivePassType::Render;
        assert!(pass_type.is_active());
        assert_eq!(pass_type.to_string(), "Render");
    }

    #[test]
    fn test_active_pass_type_compute() {
        let pass_type = ActivePassType::Compute;
        assert!(pass_type.is_active());
        assert_eq!(pass_type.to_string(), "Compute");
    }

    #[test]
    fn test_active_pass_type_default() {
        let pass_type = ActivePassType::default();
        assert_eq!(pass_type, ActivePassType::None);
        assert!(!pass_type.is_active());
    }

    #[test]
    fn test_active_pass_type_from_encoder_state() {
        assert_eq!(
            ActivePassType::from_encoder_state(EncoderState::Created),
            ActivePassType::None
        );
        assert_eq!(
            ActivePassType::from_encoder_state(EncoderState::InRenderPass),
            ActivePassType::Render
        );
        assert_eq!(
            ActivePassType::from_encoder_state(EncoderState::InComputePass),
            ActivePassType::Compute
        );
        assert_eq!(
            ActivePassType::from_encoder_state(EncoderState::Finished),
            ActivePassType::None
        );
    }

    #[test]
    fn test_active_pass_type_debug() {
        let debug_str = format!("{:?}", ActivePassType::Render);
        assert!(debug_str.contains("Render"));
    }

    #[test]
    fn test_active_pass_type_clone_copy() {
        let pass_type = ActivePassType::Compute;
        let cloned = pass_type.clone();
        let copied = pass_type;
        assert_eq!(pass_type, cloned);
        assert_eq!(pass_type, copied);
    }

    #[test]
    fn test_active_pass_type_eq_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(ActivePassType::None);
        set.insert(ActivePassType::Render);
        set.insert(ActivePassType::Compute);

        assert_eq!(set.len(), 3);
        assert!(set.contains(&ActivePassType::None));
        assert!(set.contains(&ActivePassType::Render));
        assert!(set.contains(&ActivePassType::Compute));
    }

    // -------------------------------------------------------------------------
    // PassError and PassErrorKind tests (T-WGPU-P4.1.3)
    // -------------------------------------------------------------------------

    #[test]
    fn test_pass_error_new() {
        let error = PassError::new(PassErrorKind::PassAlreadyActive);
        assert_eq!(error.kind, PassErrorKind::PassAlreadyActive);
    }

    #[test]
    fn test_pass_error_constructors() {
        assert_eq!(PassError::already_active().kind, PassErrorKind::PassAlreadyActive);
        assert_eq!(PassError::no_active().kind, PassErrorKind::NoActivePass);
        assert_eq!(PassError::wrong_type().kind, PassErrorKind::WrongPassType);
        assert_eq!(PassError::encoder_finished().kind, PassErrorKind::EncoderFinished);
    }

    #[test]
    fn test_pass_error_display() {
        let error = PassError::already_active();
        let display_str = error.to_string();
        assert!(display_str.contains("cannot begin a new pass"));
    }

    #[test]
    fn test_pass_error_kind_display() {
        // PassAlreadyActive: "cannot begin a new pass while another pass is active"
        let msg = PassErrorKind::PassAlreadyActive.to_string();
        assert!(msg.contains("pass") && msg.contains("active"), "got: {}", msg);

        // NoActivePass: "no active pass to end"
        let msg = PassErrorKind::NoActivePass.to_string();
        assert!(msg.contains("no active"), "got: {}", msg);

        // WrongPassType: "active pass is not the expected type"
        let msg = PassErrorKind::WrongPassType.to_string();
        assert!(msg.contains("not the expected"), "got: {}", msg);

        // EncoderFinished: "encoder has already been finished"
        let msg = PassErrorKind::EncoderFinished.to_string();
        assert!(msg.contains("already been finished"), "got: {}", msg);
    }

    #[test]
    fn test_pass_error_is_std_error() {
        let error = PassError::no_active();
        let _: &dyn std::error::Error = &error;
    }

    #[test]
    fn test_pass_error_debug() {
        let error = PassError::wrong_type();
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("PassError"));
        assert!(debug_str.contains("WrongPassType"));
    }

    #[test]
    fn test_pass_error_kind_clone_copy() {
        let kind = PassErrorKind::PassAlreadyActive;
        let cloned = kind.clone();
        let copied = kind;
        assert_eq!(kind, cloned);
        assert_eq!(kind, copied);
    }

    #[test]
    fn test_pass_error_clone() {
        let error = PassError::encoder_finished();
        let cloned = error.clone();
        assert_eq!(error.kind, cloned.kind);
    }

    // -------------------------------------------------------------------------
    // Pass tracking method tests (T-WGPU-P4.1.3)
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_active_pass_initial_none() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        assert_eq!(encoder.active_pass(), ActivePassType::None);
        assert!(!encoder.has_active_pass());
    }

    #[test]
    fn test_encoder_active_pass_after_render() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_render_pass_internal();
        assert_eq!(encoder.active_pass(), ActivePassType::Render);
        assert!(encoder.has_active_pass());

        encoder.end_render_pass_internal();
        assert_eq!(encoder.active_pass(), ActivePassType::None);
        assert!(!encoder.has_active_pass());
    }

    #[test]
    fn test_encoder_active_pass_after_compute() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_compute_pass_internal();
        assert_eq!(encoder.active_pass(), ActivePassType::Compute);
        assert!(encoder.has_active_pass());

        encoder.end_compute_pass_internal();
        assert_eq!(encoder.active_pass(), ActivePassType::None);
        assert!(!encoder.has_active_pass());
    }

    #[test]
    fn test_encoder_begin_render_pass_checked_success() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        let result = encoder.begin_render_pass_checked();
        assert!(result.is_ok());
        assert_eq!(encoder.state(), EncoderState::InRenderPass);
        assert_eq!(encoder.active_pass(), ActivePassType::Render);
    }

    #[test]
    fn test_encoder_begin_compute_pass_checked_success() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        let result = encoder.begin_compute_pass_checked();
        assert!(result.is_ok());
        assert_eq!(encoder.state(), EncoderState::InComputePass);
        assert_eq!(encoder.active_pass(), ActivePassType::Compute);
    }

    #[test]
    fn test_encoder_begin_pass_while_render_active() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Start a render pass
        encoder.begin_render_pass_checked().unwrap();

        // Try to start another render pass - should fail
        let result = encoder.begin_render_pass_checked();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::PassAlreadyActive);

        // Try to start a compute pass - should fail
        let result = encoder.begin_compute_pass_checked();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::PassAlreadyActive);
    }

    #[test]
    fn test_encoder_begin_pass_while_compute_active() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Start a compute pass
        encoder.begin_compute_pass_checked().unwrap();

        // Try to start a render pass - should fail
        let result = encoder.begin_render_pass_checked();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::PassAlreadyActive);

        // Try to start another compute pass - should fail
        let result = encoder.begin_compute_pass_checked();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::PassAlreadyActive);
    }

    #[test]
    fn test_encoder_end_pass_render() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_render_pass_checked().unwrap();

        let result = encoder.end_pass();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ActivePassType::Render);
        assert_eq!(encoder.state(), EncoderState::Created);
    }

    #[test]
    fn test_encoder_end_pass_compute() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_compute_pass_checked().unwrap();

        let result = encoder.end_pass();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ActivePassType::Compute);
        assert_eq!(encoder.state(), EncoderState::Created);
    }

    #[test]
    fn test_encoder_end_pass_no_active() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // No pass active
        let result = encoder.end_pass();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::NoActivePass);
    }

    #[test]
    fn test_encoder_end_pass_typed_correct() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_render_pass_checked().unwrap();

        let result = encoder.end_pass_typed(ActivePassType::Render);
        assert!(result.is_ok());
        assert_eq!(encoder.state(), EncoderState::Created);
    }

    #[test]
    fn test_encoder_end_pass_typed_wrong_type() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_render_pass_checked().unwrap();

        // Try to end as compute - should fail
        let result = encoder.end_pass_typed(ActivePassType::Compute);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::WrongPassType);

        // State should still be InRenderPass
        assert_eq!(encoder.state(), EncoderState::InRenderPass);
    }

    #[test]
    fn test_encoder_end_pass_typed_no_active() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        let result = encoder.end_pass_typed(ActivePassType::Render);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::NoActivePass);
    }

    #[test]
    fn test_encoder_finish_checked_success() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        let result = encoder.finish_checked();
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_finish_checked_with_active_pass() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.begin_render_pass_checked().unwrap();

        let result = encoder.finish_checked();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::PassAlreadyActive);
    }

    #[test]
    fn test_encoder_auto_end_pass_on_finish() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Auto End Test"),
            42,
        );

        // Start a render pass but don't end it
        encoder.begin_render_pass_checked().unwrap();

        // Finish should auto-end the pass (with warning)
        // Note: The warning is printed to stderr, not captured here
        let _cb = encoder.finish();
        // If we get here, finish succeeded despite active pass
    }

    #[test]
    fn test_encoder_auto_end_compute_pass_on_finish() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Auto End Compute Test"),
            99,
        );

        // Start a compute pass but don't end it
        encoder.begin_compute_pass_checked().unwrap();

        // Finish should auto-end the pass (with warning)
        let _cb = encoder.finish();
    }

    #[test]
    fn test_encoder_multiple_pass_cycles_checked() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Render pass cycle
        assert!(encoder.begin_render_pass_checked().is_ok());
        assert_eq!(encoder.active_pass(), ActivePassType::Render);
        assert!(encoder.end_pass().is_ok());
        assert_eq!(encoder.active_pass(), ActivePassType::None);

        // Compute pass cycle
        assert!(encoder.begin_compute_pass_checked().is_ok());
        assert_eq!(encoder.active_pass(), ActivePassType::Compute);
        assert!(encoder.end_pass().is_ok());
        assert_eq!(encoder.active_pass(), ActivePassType::None);

        // Another render pass
        assert!(encoder.begin_render_pass_checked().is_ok());
        assert!(encoder.end_pass_typed(ActivePassType::Render).is_ok());

        // Finish
        let result = encoder.finish_checked();
        assert!(result.is_ok());
    }

    #[test]
    fn test_encoder_pass_tracking_thread_safety() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Test that pass tracking methods can be called repeatedly
        for _ in 0..100 {
            let _ = encoder.active_pass();
            let _ = encoder.has_active_pass();
        }

        // Begin and end passes in a loop
        for _ in 0..10 {
            encoder.begin_render_pass_checked().unwrap();
            assert!(encoder.has_active_pass());
            encoder.end_pass().unwrap();
            assert!(!encoder.has_active_pass());
        }

        assert_eq!(encoder.active_pass(), ActivePassType::None);
    }

    #[test]
    fn test_encoder_begin_pass_after_finish_state() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Manually set finished state (simulating post-finish check)
        encoder.state.store(EncoderState::Finished as u8, Ordering::Release);

        // Should fail with EncoderFinished
        let result = encoder.begin_render_pass_checked();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::EncoderFinished);

        let result = encoder.begin_compute_pass_checked();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::EncoderFinished);
    }

    #[test]
    fn test_encoder_end_pass_after_finish_state() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        encoder.state.store(EncoderState::Finished as u8, Ordering::Release);

        let result = encoder.end_pass();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind, PassErrorKind::EncoderFinished);
    }

    // -------------------------------------------------------------------------
    // Command Buffer Finalization tests (T-WGPU-P4.1.4)
    // -------------------------------------------------------------------------

    #[test]
    fn test_into_command_buffer_success() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Into CB Test"),
            0,
        );

        // Should successfully consume encoder and return command buffer
        let result = encoder.into_command_buffer();
        assert!(result.is_ok());

        let _command_buffer: wgpu::CommandBuffer = result.unwrap();
    }

    #[test]
    fn test_into_command_buffer_with_active_render_pass() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Start a render pass
        encoder.begin_render_pass_checked().unwrap();

        // Should fail because pass is active
        let result = encoder.into_command_buffer();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.current, EncoderState::InRenderPass);
        assert_eq!(err.attempted, EncoderState::Finished);
    }

    #[test]
    fn test_into_command_buffer_with_active_compute_pass() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Start a compute pass
        encoder.begin_compute_pass_checked().unwrap();

        // Should fail because pass is active
        let result = encoder.into_command_buffer();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.current, EncoderState::InComputePass);
        assert_eq!(err.attempted, EncoderState::Finished);
    }

    #[test]
    fn test_into_command_buffer_already_finished() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Manually set to finished state
        encoder.state.store(EncoderState::Finished as u8, Ordering::Release);

        // Should fail because already finished
        let result = encoder.into_command_buffer();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.current, EncoderState::Finished);
    }

    #[test]
    fn test_submit_to_queue_single_buffer() {
        let Some(device) = test_device() else {
            return;
        };

        // Create a queue
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Submit Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create device");

        let device = Arc::new(device);

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Submit Test"),
            0,
        );

        let cb = encoder.finish();

        // Submit should succeed and return a submission index
        let _index = TrinityCommandEncoder::submit_to_queue(&queue, std::iter::once(cb));
    }

    #[test]
    fn test_submit_to_queue_multiple_buffers() {
        let Some(device) = test_device() else {
            return;
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Multi Submit Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create device");

        let device = Arc::new(device);

        let encoder1 = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 0);
        let encoder2 = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 1);
        let encoder3 = TrinityCommandEncoder::new(&device, &CommandEncoderDescriptor::new(), 2);

        let cb1 = encoder1.finish();
        let cb2 = encoder2.finish();
        let cb3 = encoder3.finish();

        // Submit multiple command buffers at once
        let _index = TrinityCommandEncoder::submit_to_queue(&queue, [cb1, cb2, cb3]);
    }

    #[test]
    fn test_finish_and_submit_success() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Finish And Submit Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create device");

        let device = Arc::new(device);

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Finish And Submit"),
            42,
        );

        // Should finish and submit successfully
        let result = encoder.finish_and_submit(&queue);
        assert!(result.is_ok());
    }

    #[test]
    fn test_finish_and_submit_with_active_pass() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Finish And Submit Active Pass Test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create device");

        let device = Arc::new(device);

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // Start a render pass but don't end it
        encoder.begin_render_pass_checked().unwrap();

        // Should fail because pass is active
        let result = encoder.finish_and_submit(&queue);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.current, EncoderState::InRenderPass);
    }

    #[test]
    fn test_finish_and_submit_lenient_success() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Lenient Submit Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create device");

        let device = Arc::new(device);

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Lenient Submit"),
            0,
        );

        // Should succeed
        let _index = encoder.finish_and_submit_lenient(&queue);
    }

    #[test]
    fn test_finish_and_submit_lenient_with_active_pass() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Lenient Submit Active Pass Test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create device");

        let device = Arc::new(device);

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new().label("Lenient With Pass"),
            99,
        );

        // Start a render pass but don't end it
        encoder.begin_render_pass_checked().unwrap();

        // Should still succeed (with warning), auto-ending the pass
        let _index = encoder.finish_and_submit_lenient(&queue);
    }

    #[test]
    fn test_into_command_buffer_consumes_encoder() {
        let Some(device) = test_device() else {
            return;
        };

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        // This should consume the encoder
        let _cb = encoder.into_command_buffer().unwrap();

        // encoder is now consumed, can't use it again
        // (This is a compile-time check, not runtime)
    }

    #[test]
    fn test_finish_and_submit_returns_submission_index() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        let Some(adapter) = adapter else {
            return;
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Submission Index Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create device");

        let device = Arc::new(device);

        let encoder = TrinityCommandEncoder::new(
            &device,
            &CommandEncoderDescriptor::new(),
            0,
        );

        let result = encoder.finish_and_submit(&queue);
        assert!(result.is_ok());

        // The SubmissionIndex type exists
        let _index: wgpu::SubmissionIndex = result.unwrap();
    }
}
