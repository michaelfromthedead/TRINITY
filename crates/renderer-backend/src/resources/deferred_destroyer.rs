//! Deferred resource destruction for TRINITY.
//!
//! This module provides deferred resource destruction to handle the case where
//! CPU code drops a resource (buffer, texture) while the GPU may still be using
//! it in an in-flight frame. Resources are queued for destruction and only
//! released after N frames have passed.
//!
//! # Architecture
//!
//! The deferred destroyer maintains a queue of pending destructions. Each entry
//! tracks the resource and the frame at which it should be destroyed. On each
//! frame end, `process_frame()` iterates the queue and drops resources whose
//! destruction frame has been reached.
//!
//! ```text
//! Frame 0: defer_buffer(buffer_a, 2) -> destroy_at_frame = 2
//! Frame 1: defer_texture(texture_b, 2) -> destroy_at_frame = 3
//! Frame 2: process_frame(2) -> drops buffer_a
//! Frame 3: process_frame(3) -> drops texture_b
//! ```
//!
//! # Default Delay
//!
//! The default delay is 2 frames, which accounts for:
//! - Frame N: Resource last used
//! - Frame N+1: GPU may still be executing frame N
//! - Frame N+2: Safe to destroy (frame N has completed)
//!
//! This aligns with typical triple-buffering (3 frames in flight) scenarios
//! where waiting 2 frames ensures all GPU work referencing the resource has
//! completed.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::deferred_destroyer::{DeferredDestroyer, DeferredResource};
//!
//! # fn example(device: &wgpu::Device) {
//! let mut destroyer = DeferredDestroyer::new();
//!
//! // Create a buffer
//! let buffer = device.create_buffer(&wgpu::BufferDescriptor {
//!     label: Some("temp_buffer"),
//!     size: 1024,
//!     usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
//!     mapped_at_creation: false,
//! });
//!
//! // Queue for destruction after current frame + 2
//! let current_frame = 10;
//! destroyer.defer_buffer(buffer, current_frame);
//!
//! // Process frame 10, 11 - nothing happens
//! destroyer.process_frame(10);
//! destroyer.process_frame(11);
//!
//! // Process frame 12 - buffer is destroyed
//! destroyer.process_frame(12);
//! # }
//! ```
//!
//! # Metrics
//!
//! The destroyer tracks metrics for monitoring:
//! - `pending_count` - Current number of pending destructions
//! - `destroyed_count` - Total resources destroyed since creation
//! - `peak_pending` - Maximum pending count observed

use log::{debug, warn};
use std::any::Any;
use wgpu::{Buffer, Texture};

// ============================================================================
// Constants
// ============================================================================

/// Default number of frames to delay resource destruction.
///
/// This value of 2 ensures safety with triple-buffering (3 frames in flight):
/// - Frame N: Resource last used by GPU
/// - Frame N+1: GPU may still be executing frame N
/// - Frame N+2: Frame N has completed, safe to destroy
pub const DEFAULT_DESTRUCTION_DELAY: u64 = 2;

// ============================================================================
// DeferredResource
// ============================================================================

/// A GPU resource queued for deferred destruction.
///
/// This enum wraps wgpu resources that need deferred destruction.
/// When dropped, the underlying resource is automatically destroyed.
///
/// # Extensibility
///
/// For resources not covered by explicit variants, use [`DeferredResource::Other`]
/// with a boxed type that implements `Any + Send`.
pub enum DeferredResource {
    /// A wgpu buffer.
    Buffer(Buffer),
    /// A wgpu texture.
    Texture(Texture),
    /// An arbitrary resource (for extensibility).
    ///
    /// This variant allows deferring destruction of any `Send` type,
    /// such as bind groups, samplers, or custom resource wrappers.
    Other(Box<dyn Any + Send>),
}

impl DeferredResource {
    /// Returns a debug label for this resource type.
    pub fn type_name(&self) -> &'static str {
        match self {
            DeferredResource::Buffer(_) => "Buffer",
            DeferredResource::Texture(_) => "Texture",
            DeferredResource::Other(_) => "Other",
        }
    }
}

impl std::fmt::Debug for DeferredResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeferredResource::Buffer(b) => {
                f.debug_struct("DeferredResource::Buffer")
                    .field("size", &b.size())
                    .finish()
            }
            DeferredResource::Texture(t) => {
                let size = t.size();
                f.debug_struct("DeferredResource::Texture")
                    .field("width", &size.width)
                    .field("height", &size.height)
                    .field("depth_or_array_layers", &size.depth_or_array_layers)
                    .finish()
            }
            DeferredResource::Other(_) => f.debug_struct("DeferredResource::Other").finish(),
        }
    }
}

// ============================================================================
// PendingDestruction
// ============================================================================

/// A resource queued for destruction at a specific frame.
struct PendingDestruction {
    /// The resource to be destroyed.
    resource: DeferredResource,
    /// The frame index at which this resource should be destroyed.
    destroy_at_frame: u64,
    /// Optional label for debugging.
    label: Option<String>,
}

impl std::fmt::Debug for PendingDestruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingDestruction")
            .field("resource", &self.resource.type_name())
            .field("destroy_at_frame", &self.destroy_at_frame)
            .field("label", &self.label)
            .finish()
    }
}

// ============================================================================
// DeferredDestroyerMetrics
// ============================================================================

/// Metrics tracking for the deferred destroyer.
#[derive(Debug, Clone, Default)]
pub struct DeferredDestroyerMetrics {
    /// Current number of pending destructions.
    pub pending_count: usize,
    /// Total resources destroyed since creation.
    pub destroyed_count: u64,
    /// Maximum pending count ever observed.
    pub peak_pending: usize,
    /// Total buffers destroyed.
    pub buffers_destroyed: u64,
    /// Total textures destroyed.
    pub textures_destroyed: u64,
    /// Total other resources destroyed.
    pub other_destroyed: u64,
}

// ============================================================================
// DeferredDestroyer
// ============================================================================

/// Deferred resource destruction manager.
///
/// Queues GPU resources for destruction after a configurable number of frames
/// have passed, ensuring the GPU has finished using them.
///
/// # Thread Safety
///
/// `DeferredDestroyer` is not `Sync` by default. For concurrent access,
/// wrap in `Arc<Mutex<DeferredDestroyer>>` or use per-thread instances.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::deferred_destroyer::DeferredDestroyer;
///
/// # fn example(device: &wgpu::Device) {
/// let mut destroyer = DeferredDestroyer::new();
///
/// // At frame 10, queue a buffer for destruction
/// let buffer = device.create_buffer(&wgpu::BufferDescriptor {
///     label: Some("temp"),
///     size: 256,
///     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
///     mapped_at_creation: false,
/// });
/// destroyer.defer_buffer(buffer, 10);
///
/// // Process each frame
/// for frame in 10..=12 {
///     destroyer.process_frame(frame);
/// }
/// // Buffer destroyed after frame 12
/// # }
/// ```
pub struct DeferredDestroyer {
    /// Queue of pending destructions.
    pending: Vec<PendingDestruction>,
    /// Default frame delay for destruction.
    default_delay: u64,
    /// Total resources destroyed since creation.
    destroyed_count: u64,
    /// Peak pending count.
    peak_pending: usize,
    /// Buffers destroyed count.
    buffers_destroyed: u64,
    /// Textures destroyed count.
    textures_destroyed: u64,
    /// Other resources destroyed count.
    other_destroyed: u64,
}

impl DeferredDestroyer {
    /// Create a new deferred destroyer with the default delay.
    pub fn new() -> Self {
        Self::with_delay(DEFAULT_DESTRUCTION_DELAY)
    }

    /// Create a new deferred destroyer with a custom default delay.
    ///
    /// # Arguments
    ///
    /// * `delay` - Number of frames to wait before destroying resources
    pub fn with_delay(delay: u64) -> Self {
        Self {
            pending: Vec::new(),
            default_delay: delay,
            destroyed_count: 0,
            peak_pending: 0,
            buffers_destroyed: 0,
            textures_destroyed: 0,
            other_destroyed: 0,
        }
    }

    /// Queue a buffer for deferred destruction.
    ///
    /// The buffer will be destroyed when `process_frame(current_frame + default_delay)`
    /// is called.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to destroy
    /// * `current_frame` - The current frame index
    pub fn defer_buffer(&mut self, buffer: Buffer, current_frame: u64) {
        self.defer_buffer_with_delay(buffer, current_frame, self.default_delay);
    }

    /// Queue a buffer for deferred destruction with a custom delay.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to destroy
    /// * `current_frame` - The current frame index
    /// * `frame_delay` - Number of frames to wait before destruction
    pub fn defer_buffer_with_delay(&mut self, buffer: Buffer, current_frame: u64, frame_delay: u64) {
        let destroy_at_frame = current_frame + frame_delay;
        debug!(
            "Deferring buffer destruction: size={}, destroy_at_frame={}",
            buffer.size(),
            destroy_at_frame
        );
        self.pending.push(PendingDestruction {
            resource: DeferredResource::Buffer(buffer),
            destroy_at_frame,
            label: None,
        });
        self.update_peak();
    }

    /// Queue a buffer for deferred destruction with a label.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to destroy
    /// * `current_frame` - The current frame index
    /// * `label` - Debug label for this destruction
    pub fn defer_buffer_labeled(
        &mut self,
        buffer: Buffer,
        current_frame: u64,
        label: impl Into<String>,
    ) {
        let destroy_at_frame = current_frame + self.default_delay;
        let label_str = label.into();
        debug!(
            "Deferring buffer destruction: label={}, size={}, destroy_at_frame={}",
            label_str,
            buffer.size(),
            destroy_at_frame
        );
        self.pending.push(PendingDestruction {
            resource: DeferredResource::Buffer(buffer),
            destroy_at_frame,
            label: Some(label_str),
        });
        self.update_peak();
    }

    /// Queue a texture for deferred destruction.
    ///
    /// The texture will be destroyed when `process_frame(current_frame + default_delay)`
    /// is called.
    ///
    /// # Arguments
    ///
    /// * `texture` - The texture to destroy
    /// * `current_frame` - The current frame index
    pub fn defer_texture(&mut self, texture: Texture, current_frame: u64) {
        self.defer_texture_with_delay(texture, current_frame, self.default_delay);
    }

    /// Queue a texture for deferred destruction with a custom delay.
    ///
    /// # Arguments
    ///
    /// * `texture` - The texture to destroy
    /// * `current_frame` - The current frame index
    /// * `frame_delay` - Number of frames to wait before destruction
    pub fn defer_texture_with_delay(
        &mut self,
        texture: Texture,
        current_frame: u64,
        frame_delay: u64,
    ) {
        let destroy_at_frame = current_frame + frame_delay;
        let size = texture.size();
        debug!(
            "Deferring texture destruction: {}x{}, destroy_at_frame={}",
            size.width, size.height, destroy_at_frame
        );
        self.pending.push(PendingDestruction {
            resource: DeferredResource::Texture(texture),
            destroy_at_frame,
            label: None,
        });
        self.update_peak();
    }

    /// Queue a texture for deferred destruction with a label.
    ///
    /// # Arguments
    ///
    /// * `texture` - The texture to destroy
    /// * `current_frame` - The current frame index
    /// * `label` - Debug label for this destruction
    pub fn defer_texture_labeled(
        &mut self,
        texture: Texture,
        current_frame: u64,
        label: impl Into<String>,
    ) {
        let destroy_at_frame = current_frame + self.default_delay;
        let label_str = label.into();
        let size = texture.size();
        debug!(
            "Deferring texture destruction: label={}, {}x{}, destroy_at_frame={}",
            label_str, size.width, size.height, destroy_at_frame
        );
        self.pending.push(PendingDestruction {
            resource: DeferredResource::Texture(texture),
            destroy_at_frame,
            label: Some(label_str),
        });
        self.update_peak();
    }

    /// Queue an arbitrary resource for deferred destruction.
    ///
    /// This allows deferring destruction of any `Send` type, such as
    /// bind groups, samplers, or custom resource wrappers.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource to destroy (must be `Send`)
    /// * `current_frame` - The current frame index
    pub fn defer_other<T: Any + Send + 'static>(&mut self, resource: T, current_frame: u64) {
        self.defer_other_with_delay(resource, current_frame, self.default_delay);
    }

    /// Queue an arbitrary resource for deferred destruction with a custom delay.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource to destroy (must be `Send`)
    /// * `current_frame` - The current frame index
    /// * `frame_delay` - Number of frames to wait before destruction
    pub fn defer_other_with_delay<T: Any + Send + 'static>(
        &mut self,
        resource: T,
        current_frame: u64,
        frame_delay: u64,
    ) {
        let destroy_at_frame = current_frame + frame_delay;
        debug!(
            "Deferring other resource destruction: type={}, destroy_at_frame={}",
            std::any::type_name::<T>(),
            destroy_at_frame
        );
        self.pending.push(PendingDestruction {
            resource: DeferredResource::Other(Box::new(resource)),
            destroy_at_frame,
            label: None,
        });
        self.update_peak();
    }

    /// Queue an arbitrary resource for deferred destruction with a label.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource to destroy (must be `Send`)
    /// * `current_frame` - The current frame index
    /// * `label` - Debug label for this destruction
    pub fn defer_other_labeled<T: Any + Send + 'static>(
        &mut self,
        resource: T,
        current_frame: u64,
        label: impl Into<String>,
    ) {
        let destroy_at_frame = current_frame + self.default_delay;
        let label_str = label.into();
        debug!(
            "Deferring other resource destruction: label={}, type={}, destroy_at_frame={}",
            label_str,
            std::any::type_name::<T>(),
            destroy_at_frame
        );
        self.pending.push(PendingDestruction {
            resource: DeferredResource::Other(Box::new(resource)),
            destroy_at_frame,
            label: Some(label_str),
        });
        self.update_peak();
    }

    /// Queue a pre-wrapped deferred resource for destruction.
    ///
    /// This is useful when you already have a `DeferredResource` enum value.
    ///
    /// # Arguments
    ///
    /// * `resource` - The wrapped resource to destroy
    /// * `current_frame` - The current frame index
    pub fn defer_resource(&mut self, resource: DeferredResource, current_frame: u64) {
        self.defer_resource_with_delay(resource, current_frame, self.default_delay);
    }

    /// Queue a pre-wrapped deferred resource for destruction with a custom delay.
    ///
    /// # Arguments
    ///
    /// * `resource` - The wrapped resource to destroy
    /// * `current_frame` - The current frame index
    /// * `frame_delay` - Number of frames to wait before destruction
    pub fn defer_resource_with_delay(
        &mut self,
        resource: DeferredResource,
        current_frame: u64,
        frame_delay: u64,
    ) {
        let destroy_at_frame = current_frame + frame_delay;
        debug!(
            "Deferring resource destruction: type={}, destroy_at_frame={}",
            resource.type_name(),
            destroy_at_frame
        );
        self.pending.push(PendingDestruction {
            resource,
            destroy_at_frame,
            label: None,
        });
        self.update_peak();
    }

    /// Process a frame, destroying all resources scheduled for this frame or earlier.
    ///
    /// This should be called at the end of each frame, after all GPU work
    /// for the frame has been submitted.
    ///
    /// # Arguments
    ///
    /// * `current_frame` - The current frame index
    ///
    /// # Returns
    ///
    /// The number of resources destroyed this frame.
    pub fn process_frame(&mut self, current_frame: u64) -> usize {
        let initial_count = self.pending.len();

        // Partition: keep resources that should NOT be destroyed yet
        let mut destroyed_this_frame = 0;
        self.pending.retain(|pending| {
            if pending.destroy_at_frame <= current_frame {
                // Resource is ready to be destroyed
                match &pending.resource {
                    DeferredResource::Buffer(b) => {
                        debug!(
                            "Destroying deferred buffer: size={}, label={:?}",
                            b.size(),
                            pending.label
                        );
                        self.buffers_destroyed += 1;
                    }
                    DeferredResource::Texture(t) => {
                        let size = t.size();
                        debug!(
                            "Destroying deferred texture: {}x{}, label={:?}",
                            size.width, size.height, pending.label
                        );
                        self.textures_destroyed += 1;
                    }
                    DeferredResource::Other(_) => {
                        debug!("Destroying deferred other resource: label={:?}", pending.label);
                        self.other_destroyed += 1;
                    }
                }
                destroyed_this_frame += 1;
                self.destroyed_count += 1;
                false // Remove from queue (will be dropped)
            } else {
                true // Keep in queue
            }
        });

        if destroyed_this_frame > 0 {
            debug!(
                "Frame {}: destroyed {} resources, {} remaining",
                current_frame,
                destroyed_this_frame,
                self.pending.len()
            );
        }

        destroyed_this_frame
    }

    /// Get the current number of pending destructions.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if there are any pending destructions.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Get the default destruction delay.
    pub fn default_delay(&self) -> u64 {
        self.default_delay
    }

    /// Set the default destruction delay.
    ///
    /// This only affects future `defer_*` calls, not already queued resources.
    pub fn set_default_delay(&mut self, delay: u64) {
        self.default_delay = delay;
    }

    /// Get current metrics.
    pub fn metrics(&self) -> DeferredDestroyerMetrics {
        DeferredDestroyerMetrics {
            pending_count: self.pending.len(),
            destroyed_count: self.destroyed_count,
            peak_pending: self.peak_pending,
            buffers_destroyed: self.buffers_destroyed,
            textures_destroyed: self.textures_destroyed,
            other_destroyed: self.other_destroyed,
        }
    }

    /// Clear all pending destructions immediately.
    ///
    /// # Warning
    ///
    /// This destroys all queued resources immediately, which may cause GPU
    /// validation errors if the GPU is still using them. Only use this
    /// when shutting down and you've ensured all GPU work has completed.
    pub fn clear(&mut self) {
        let count = self.pending.len();
        if count > 0 {
            warn!(
                "Clearing {} pending resources immediately (potential GPU hazard)",
                count
            );
        }
        // Update counts before clearing
        for pending in &self.pending {
            match &pending.resource {
                DeferredResource::Buffer(_) => self.buffers_destroyed += 1,
                DeferredResource::Texture(_) => self.textures_destroyed += 1,
                DeferredResource::Other(_) => self.other_destroyed += 1,
            }
            self.destroyed_count += 1;
        }
        self.pending.clear();
    }

    /// Update peak pending count.
    fn update_peak(&mut self) {
        if self.pending.len() > self.peak_pending {
            self.peak_pending = self.pending.len();
        }
    }
}

impl Default for DeferredDestroyer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DeferredDestroyer {
    fn drop(&mut self) {
        if !self.pending.is_empty() {
            warn!(
                "DeferredDestroyer dropped with {} pending resources - destroying now",
                self.pending.len()
            );
            // Resources will be dropped when pending Vec is dropped
            // We just log the warning; no special action needed
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a mock buffer for testing
    // In real tests with wgpu, you'd use pollster and create actual resources
    // For unit tests, we test the destroyer logic without actual GPU resources

    /// Test creating a deferred destroyer with default settings.
    #[test]
    fn test_new_destroyer() {
        let destroyer = DeferredDestroyer::new();
        assert_eq!(destroyer.default_delay(), DEFAULT_DESTRUCTION_DELAY);
        assert!(destroyer.is_empty());
        assert_eq!(destroyer.pending_count(), 0);
    }

    /// Test creating a deferred destroyer with custom delay.
    #[test]
    fn test_custom_delay() {
        let destroyer = DeferredDestroyer::with_delay(5);
        assert_eq!(destroyer.default_delay(), 5);
        assert!(destroyer.is_empty());
    }

    /// Test setting the default delay.
    #[test]
    fn test_set_default_delay() {
        let mut destroyer = DeferredDestroyer::new();
        assert_eq!(destroyer.default_delay(), DEFAULT_DESTRUCTION_DELAY);

        destroyer.set_default_delay(10);
        assert_eq!(destroyer.default_delay(), 10);
    }

    /// Test metrics on empty destroyer.
    #[test]
    fn test_metrics_empty() {
        let destroyer = DeferredDestroyer::new();
        let metrics = destroyer.metrics();

        assert_eq!(metrics.pending_count, 0);
        assert_eq!(metrics.destroyed_count, 0);
        assert_eq!(metrics.peak_pending, 0);
        assert_eq!(metrics.buffers_destroyed, 0);
        assert_eq!(metrics.textures_destroyed, 0);
        assert_eq!(metrics.other_destroyed, 0);
    }

    /// Test deferring an "other" resource type.
    #[test]
    fn test_defer_other_resource() {
        let mut destroyer = DeferredDestroyer::new();

        // Use a simple type as "other" resource
        destroyer.defer_other(String::from("test_resource"), 0);

        assert_eq!(destroyer.pending_count(), 1);
        assert!(!destroyer.is_empty());
    }

    /// Test deferring with custom delay.
    #[test]
    fn test_defer_other_with_custom_delay() {
        let mut destroyer = DeferredDestroyer::new();

        destroyer.defer_other_with_delay(42u32, 0, 5);

        assert_eq!(destroyer.pending_count(), 1);

        // Should not destroy at frame 4
        let destroyed = destroyer.process_frame(4);
        assert_eq!(destroyed, 0);
        assert_eq!(destroyer.pending_count(), 1);

        // Should destroy at frame 5
        let destroyed = destroyer.process_frame(5);
        assert_eq!(destroyed, 1);
        assert!(destroyer.is_empty());
    }

    /// Test process_frame destroys correct resources.
    #[test]
    fn test_process_frame_timing() {
        let mut destroyer = DeferredDestroyer::with_delay(2);

        // Queue resources at different frames
        destroyer.defer_other("resource_a".to_string(), 10); // destroy at 12
        destroyer.defer_other("resource_b".to_string(), 11); // destroy at 13
        destroyer.defer_other("resource_c".to_string(), 12); // destroy at 14

        assert_eq!(destroyer.pending_count(), 3);

        // Frame 11: nothing destroyed yet
        let destroyed = destroyer.process_frame(11);
        assert_eq!(destroyed, 0);
        assert_eq!(destroyer.pending_count(), 3);

        // Frame 12: resource_a destroyed
        let destroyed = destroyer.process_frame(12);
        assert_eq!(destroyed, 1);
        assert_eq!(destroyer.pending_count(), 2);

        // Frame 13: resource_b destroyed
        let destroyed = destroyer.process_frame(13);
        assert_eq!(destroyed, 1);
        assert_eq!(destroyer.pending_count(), 1);

        // Frame 14: resource_c destroyed
        let destroyed = destroyer.process_frame(14);
        assert_eq!(destroyed, 1);
        assert!(destroyer.is_empty());
    }

    /// Test multiple frames in sequence.
    #[test]
    fn test_multiple_frames_sequence() {
        let mut destroyer = DeferredDestroyer::with_delay(2);

        // Queue several resources at frame 0
        for i in 0..5 {
            destroyer.defer_other(format!("resource_{}", i), 0);
        }

        assert_eq!(destroyer.pending_count(), 5);
        assert_eq!(destroyer.metrics().peak_pending, 5);

        // Process frame 0 and 1 - nothing destroyed
        assert_eq!(destroyer.process_frame(0), 0);
        assert_eq!(destroyer.process_frame(1), 0);
        assert_eq!(destroyer.pending_count(), 5);

        // Process frame 2 - all 5 destroyed
        let destroyed = destroyer.process_frame(2);
        assert_eq!(destroyed, 5);
        assert!(destroyer.is_empty());

        let metrics = destroyer.metrics();
        assert_eq!(metrics.destroyed_count, 5);
        assert_eq!(metrics.other_destroyed, 5);
    }

    /// Test default delay behavior.
    #[test]
    fn test_default_delay_behavior() {
        let mut destroyer = DeferredDestroyer::new();
        assert_eq!(destroyer.default_delay(), DEFAULT_DESTRUCTION_DELAY);

        destroyer.defer_other("test".to_string(), 100);

        // Should not destroy before delay
        assert_eq!(destroyer.process_frame(100), 0);
        assert_eq!(destroyer.process_frame(101), 0);

        // Should destroy at frame 100 + DEFAULT_DESTRUCTION_DELAY
        let expected_destroy_frame = 100 + DEFAULT_DESTRUCTION_DELAY;
        assert_eq!(destroyer.process_frame(expected_destroy_frame), 1);
    }

    /// Test cleanup on drop logs warning.
    #[test]
    fn test_cleanup_on_drop() {
        // This test verifies that Drop is implemented correctly
        // The warning is logged but we can't easily verify log output
        let mut destroyer = DeferredDestroyer::new();
        destroyer.defer_other("pending_resource".to_string(), 0);

        assert_eq!(destroyer.pending_count(), 1);
        // Destroyer will be dropped here and should log a warning
    }

    /// Test metrics tracking.
    #[test]
    fn test_metrics_tracking() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        // Queue 3 resources
        destroyer.defer_other("a".to_string(), 0);
        destroyer.defer_other("b".to_string(), 0);
        destroyer.defer_other("c".to_string(), 0);

        let metrics = destroyer.metrics();
        assert_eq!(metrics.pending_count, 3);
        assert_eq!(metrics.peak_pending, 3);
        assert_eq!(metrics.destroyed_count, 0);

        // Destroy all
        destroyer.process_frame(1);

        let metrics = destroyer.metrics();
        assert_eq!(metrics.pending_count, 0);
        assert_eq!(metrics.peak_pending, 3); // Peak should remain
        assert_eq!(metrics.destroyed_count, 3);
        assert_eq!(metrics.other_destroyed, 3);
    }

    /// Test clear method.
    #[test]
    fn test_clear() {
        let mut destroyer = DeferredDestroyer::with_delay(10);

        destroyer.defer_other("a".to_string(), 0);
        destroyer.defer_other("b".to_string(), 0);
        assert_eq!(destroyer.pending_count(), 2);

        destroyer.clear();

        assert!(destroyer.is_empty());
        let metrics = destroyer.metrics();
        assert_eq!(metrics.destroyed_count, 2);
        assert_eq!(metrics.other_destroyed, 2);
    }

    /// Test DeferredResource type_name method.
    #[test]
    fn test_deferred_resource_type_name() {
        let other = DeferredResource::Other(Box::new("test"));
        assert_eq!(other.type_name(), "Other");
    }

    /// Test DeferredResource Debug implementation.
    #[test]
    fn test_deferred_resource_debug() {
        let other = DeferredResource::Other(Box::new(42u32));
        let debug_str = format!("{:?}", other);
        assert!(debug_str.contains("Other"));
    }

    /// Test labeled deferral.
    #[test]
    fn test_labeled_deferral() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        destroyer.defer_other_labeled("resource".to_string(), 0, "my_label");

        assert_eq!(destroyer.pending_count(), 1);

        // Destroy and verify it was destroyed
        let destroyed = destroyer.process_frame(1);
        assert_eq!(destroyed, 1);
    }

    /// Test defer_resource method.
    #[test]
    fn test_defer_resource() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        let resource = DeferredResource::Other(Box::new("test_value"));
        destroyer.defer_resource(resource, 0);

        assert_eq!(destroyer.pending_count(), 1);

        destroyer.process_frame(1);
        assert!(destroyer.is_empty());
    }

    /// Test defer_resource_with_delay method.
    #[test]
    fn test_defer_resource_with_delay() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        let resource = DeferredResource::Other(Box::new("test_value"));
        destroyer.defer_resource_with_delay(resource, 0, 5);

        // Should not destroy before delay
        destroyer.process_frame(4);
        assert_eq!(destroyer.pending_count(), 1);

        // Should destroy at frame 5
        destroyer.process_frame(5);
        assert!(destroyer.is_empty());
    }

    /// Test peak pending tracks correctly across multiple operations.
    #[test]
    fn test_peak_pending_tracking() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        // Add 3 resources
        destroyer.defer_other(1, 0);
        destroyer.defer_other(2, 0);
        destroyer.defer_other(3, 0);
        assert_eq!(destroyer.metrics().peak_pending, 3);

        // Destroy all
        destroyer.process_frame(1);
        assert_eq!(destroyer.metrics().peak_pending, 3); // Peak unchanged

        // Add 2 more
        destroyer.defer_other(4, 1);
        destroyer.defer_other(5, 1);
        assert_eq!(destroyer.metrics().peak_pending, 3); // Peak still 3

        // Add 2 more (total 4)
        destroyer.defer_other(6, 1);
        destroyer.defer_other(7, 1);
        assert_eq!(destroyer.metrics().peak_pending, 4); // New peak
    }

    /// Test process_frame with later frame destroys all earlier pending.
    #[test]
    fn test_process_frame_destroys_all_earlier() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        // Queue at frames 0, 1, 2
        destroyer.defer_other("a", 0); // destroy at 1
        destroyer.defer_other("b", 1); // destroy at 2
        destroyer.defer_other("c", 2); // destroy at 3

        // Jump to frame 10 - should destroy all 3
        let destroyed = destroyer.process_frame(10);
        assert_eq!(destroyed, 3);
        assert!(destroyer.is_empty());
    }
}
