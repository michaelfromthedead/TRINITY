//! Depth Reconstruction and Resource Transitions (T-DEMO-6.5, T-DEMO-6.6)
//!
//! This module implements:
//! - **T-DEMO-6.5**: Depth reconstruction from ray march hit distances
//! - **T-DEMO-6.6**: Resource transition barriers for demoscene render passes
//!
//! # Depth Reconstruction (T-DEMO-6.5)
//!
//! When ray marching produces a hit, the linear eye-space distance must be
//! converted to a depth-buffer-compatible NDC value. This enables subsequent
//! rasterization passes to depth-test against SDF geometry.
//!
//! The conversion formula for a reversed depth buffer (near=1, far=0):
//! ```text
//! z_ndc = (far + near) / (far - near) - (2 * far * near) / (z_linear * (far - near))
//! ```
//!
//! For standard depth (near=0, far=1):
//! ```text
//! z_ndc = (far * near) / (z_linear * (near - far)) + far / (far - near)
//! ```
//!
//! # Resource Transitions (T-DEMO-6.6)
//!
//! The demoscene S13 compute pass writes to:
//! - Color buffer (storage texture, write-only)
//! - Depth buffer (storage texture, write-only)
//!
//! Subsequent rasterization passes require:
//! - Color buffer as render attachment (read or blend)
//! - Depth buffer as depth attachment (read/write)
//!
//! This module tracks resource states and inserts minimal barriers (target: 1-2
//! per frame) to ensure correct synchronization.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Depth Projection Parameters
// ---------------------------------------------------------------------------

/// Parameters for depth projection transformations.
///
/// Encapsulates near/far clip distances and provides conversion methods
/// between linear eye-space depth and NDC depth values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthProjection {
    /// Near clip plane distance (positive, in eye-space units).
    pub near: f32,
    /// Far clip plane distance (positive, in eye-space units).
    pub far: f32,
    /// Whether to use reversed-Z depth (near=1, far=0).
    /// Reversed-Z provides better precision distribution for large scenes.
    pub reversed_z: bool,
}

impl Default for DepthProjection {
    fn default() -> Self {
        Self {
            near: 0.1,
            far: 1000.0,
            reversed_z: true, // Modern default for better precision
        }
    }
}

impl DepthProjection {
    /// Create a new depth projection with custom near/far planes.
    pub fn new(near: f32, far: f32) -> Self {
        let clamped_near = near.max(0.0001); // Prevent divide-by-zero
        Self {
            near: clamped_near,
            far: far.max(clamped_near + 0.001),
            reversed_z: true,
        }
    }

    /// Create a depth projection with standard (non-reversed) depth range.
    pub fn standard(near: f32, far: f32) -> Self {
        let clamped_near = near.max(0.0001);
        Self {
            near: clamped_near,
            far: far.max(clamped_near + 0.001),
            reversed_z: false,
        }
    }

    /// Create a depth projection with reversed-Z depth range.
    pub fn reversed(near: f32, far: f32) -> Self {
        let clamped_near = near.max(0.0001);
        Self {
            near: clamped_near,
            far: far.max(clamped_near + 0.001),
            reversed_z: true,
        }
    }

    /// Convert linear eye-space depth to NDC depth.
    ///
    /// # Arguments
    ///
    /// * `z_linear` - Linear eye-space distance (positive, away from camera)
    ///
    /// # Returns
    ///
    /// NDC depth value in range [0, 1]:
    /// - Reversed-Z: near=1.0, far=0.0
    /// - Standard: near=0.0, far=1.0
    ///
    /// # Formula
    ///
    /// For reversed-Z (perspective projection):
    /// ```text
    /// z_ndc = near / z_linear
    /// ```
    ///
    /// For standard depth:
    /// ```text
    /// z_ndc = (far * (z_linear - near)) / (z_linear * (far - near))
    /// ```
    #[inline]
    pub fn linear_to_ndc(&self, z_linear: f32) -> f32 {
        if z_linear <= 0.0 {
            // Behind camera - return near clip value
            return if self.reversed_z { 1.0 } else { 0.0 };
        }

        if self.reversed_z {
            // Reversed-Z: z_ndc = near / z_linear (clamped to [0, 1])
            // When z_linear = near, z_ndc = 1.0
            // When z_linear = far,  z_ndc = near/far (close to 0)
            (self.near / z_linear).clamp(0.0, 1.0)
        } else {
            // Standard depth projection
            // z_ndc = (far * (z - near)) / (z * (far - near))
            let range = self.far - self.near;
            if range <= 0.0 {
                return 0.0;
            }
            let z_clamped = z_linear.clamp(self.near, self.far);
            let ndc = (self.far * (z_clamped - self.near)) / (z_clamped * range);
            ndc.clamp(0.0, 1.0)
        }
    }

    /// Convert NDC depth back to linear eye-space depth.
    ///
    /// # Arguments
    ///
    /// * `z_ndc` - NDC depth value in range [0, 1]
    ///
    /// # Returns
    ///
    /// Linear eye-space distance (positive, away from camera)
    #[inline]
    pub fn ndc_to_linear(&self, z_ndc: f32) -> f32 {
        if self.reversed_z {
            // Inverse of z_ndc = near / z_linear
            // z_linear = near / z_ndc
            if z_ndc <= 0.0 {
                return self.far; // At or beyond far plane
            }
            (self.near / z_ndc).clamp(self.near, self.far)
        } else {
            // Inverse of standard projection
            // z_ndc = (far * (z - near)) / (z * (far - near))
            // z_ndc * z * (far - near) = far * (z - near)
            // z_ndc * z * (far - near) = far * z - far * near
            // z * (z_ndc * (far - near) - far) = -far * near
            // z = (far * near) / (far - z_ndc * (far - near))
            if z_ndc >= 1.0 {
                return self.far;
            }
            if z_ndc <= 0.0 {
                return self.near;
            }
            let range = self.far - self.near;
            let denom = self.far - z_ndc * range;
            if denom <= 0.0 {
                return self.far;
            }
            ((self.far * self.near) / denom).clamp(self.near, self.far)
        }
    }

    /// Linearize a depth buffer value to view-space Z (for deferred rendering).
    ///
    /// This is useful for reconstructing world-space positions from depth.
    #[inline]
    pub fn linearize_depth(&self, z_ndc: f32) -> f32 {
        self.ndc_to_linear(z_ndc)
    }

    /// Pack depth reconstruction parameters for GPU upload.
    ///
    /// Returns `[near, far, near * far, far - near]` for efficient GPU computation.
    #[inline]
    pub fn to_gpu_params(&self) -> [f32; 4] {
        [
            self.near,
            self.far,
            self.near * self.far,
            self.far - self.near,
        ]
    }
}

// ---------------------------------------------------------------------------
// Depth Reconstruction Result
// ---------------------------------------------------------------------------

/// Result of ray march depth reconstruction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthReconstructionResult {
    /// Linear eye-space distance to hit.
    pub linear_depth: f32,
    /// NDC depth value for depth buffer.
    pub ndc_depth: f32,
    /// Whether the ray hit geometry (vs. sky/miss).
    pub hit: bool,
    /// Material ID at hit point (if hit).
    pub material_id: u32,
}

impl DepthReconstructionResult {
    /// Create a miss result (ray did not hit geometry).
    pub fn miss() -> Self {
        Self {
            linear_depth: f32::INFINITY,
            ndc_depth: 0.0, // Far plane for reversed-Z
            hit: false,
            material_id: 0,
        }
    }

    /// Create a hit result with depth reconstruction.
    pub fn hit(linear_depth: f32, projection: &DepthProjection, material_id: u32) -> Self {
        Self {
            linear_depth,
            ndc_depth: projection.linear_to_ndc(linear_depth),
            hit: true,
            material_id,
        }
    }

    /// Create from raw NDC depth (for comparison with raster depth).
    pub fn from_ndc(ndc_depth: f32, projection: &DepthProjection) -> Self {
        let linear = projection.ndc_to_linear(ndc_depth);
        Self {
            linear_depth: linear,
            ndc_depth,
            hit: ndc_depth > 0.0 && ndc_depth < 1.0,
            material_id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Resource State Tracking
// ---------------------------------------------------------------------------

/// GPU resource usage state for barrier scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DemoResourceState {
    /// Resource has not been used yet this frame.
    Uninitialized,
    /// Compute shader storage write (SDF ray march output).
    ComputeWrite,
    /// Compute shader storage read.
    ComputeRead,
    /// Render pass color attachment write.
    ColorAttachmentWrite,
    /// Render pass color attachment read (blending).
    ColorAttachmentRead,
    /// Render pass depth attachment write.
    DepthAttachmentWrite,
    /// Render pass depth attachment read-only.
    DepthAttachmentReadOnly,
    /// Shader resource view (texture sampling).
    ShaderRead,
    /// Copy source.
    TransferSrc,
    /// Copy destination.
    TransferDst,
    /// Present to swapchain.
    Present,
}

impl DemoResourceState {
    /// Returns true if this state allows concurrent reads.
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            Self::ComputeRead
                | Self::ColorAttachmentRead
                | Self::DepthAttachmentReadOnly
                | Self::ShaderRead
                | Self::TransferSrc
        )
    }

    /// Returns true if this state involves writes.
    pub fn is_write(&self) -> bool {
        matches!(
            self,
            Self::ComputeWrite
                | Self::ColorAttachmentWrite
                | Self::DepthAttachmentWrite
                | Self::TransferDst
        )
    }

    /// Returns true if a barrier is needed to transition from `self` to `next`.
    pub fn needs_barrier_to(&self, next: DemoResourceState) -> bool {
        if *self == next {
            // Same state - no barrier needed (may still need execution dependency)
            return false;
        }

        // Always need barrier from write to any different state
        if self.is_write() {
            return true;
        }

        // Need barrier from read to write
        if self.is_read_only() && next.is_write() {
            return true;
        }

        // Read to different read - may need layout transition
        // (conservative: barrier if states differ)
        true
    }
}

impl Default for DemoResourceState {
    fn default() -> Self {
        Self::Uninitialized
    }
}

// ---------------------------------------------------------------------------
// Resource Transition
// ---------------------------------------------------------------------------

/// A resource state transition (barrier).
#[derive(Clone, Debug, PartialEq)]
pub struct DemoResourceTransition {
    /// Unique identifier for the resource.
    pub resource_id: u32,
    /// Human-readable resource name (for debugging).
    pub name: String,
    /// State before the transition.
    pub before: DemoResourceState,
    /// State after the transition.
    pub after: DemoResourceState,
    /// Whether this is a texture (vs. buffer) resource.
    pub is_texture: bool,
}

impl DemoResourceTransition {
    /// Create a new resource transition.
    pub fn new(
        resource_id: u32,
        name: impl Into<String>,
        before: DemoResourceState,
        after: DemoResourceState,
        is_texture: bool,
    ) -> Self {
        Self {
            resource_id,
            name: name.into(),
            before,
            after,
            is_texture,
        }
    }

    /// Returns true if this transition requires an actual GPU barrier.
    pub fn requires_barrier(&self) -> bool {
        self.before.needs_barrier_to(self.after)
    }
}

impl std::fmt::Display for DemoResourceTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}[{}]: {:?} -> {:?}",
            self.name, self.resource_id, self.before, self.after
        )
    }
}

// ---------------------------------------------------------------------------
// Barrier Scheduler
// ---------------------------------------------------------------------------

/// Tracks resource states and schedules minimal barriers.
///
/// The demoscene renderer targets 1-2 barriers per frame by:
/// - Batching all S13 compute outputs into a single barrier
/// - Batching all rasterization inputs into a single barrier
#[derive(Debug, Default)]
pub struct DemoBarrierScheduler {
    /// Current state of each tracked resource.
    states: HashMap<u32, (String, DemoResourceState, bool)>,
    /// Pending transitions to batch.
    pending: Vec<DemoResourceTransition>,
    /// Total barriers issued this frame.
    barriers_this_frame: u32,
    /// Whether in the middle of a barrier batch.
    in_batch: bool,
}

impl DemoBarrierScheduler {
    /// Create a new barrier scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a resource with its initial state.
    pub fn register_resource(
        &mut self,
        resource_id: u32,
        name: impl Into<String>,
        initial_state: DemoResourceState,
        is_texture: bool,
    ) {
        self.states
            .insert(resource_id, (name.into(), initial_state, is_texture));
    }

    /// Get the current state of a resource.
    pub fn get_state(&self, resource_id: u32) -> Option<DemoResourceState> {
        self.states.get(&resource_id).map(|(_, state, _)| *state)
    }

    /// Transition a resource to a new state.
    ///
    /// Returns `Some(transition)` if a barrier is needed.
    pub fn transition(
        &mut self,
        resource_id: u32,
        new_state: DemoResourceState,
    ) -> Option<DemoResourceTransition> {
        let (name, current_state, is_texture) = self.states.get(&resource_id)?;

        if !current_state.needs_barrier_to(new_state) {
            // No barrier needed, just update state
            self.states
                .insert(resource_id, (name.clone(), new_state, *is_texture));
            return None;
        }

        let transition = DemoResourceTransition::new(
            resource_id,
            name.clone(),
            *current_state,
            new_state,
            *is_texture,
        );

        // Update state
        self.states
            .insert(resource_id, (name.clone(), new_state, *is_texture));

        Some(transition)
    }

    /// Begin a barrier batch (for batching multiple transitions).
    pub fn begin_batch(&mut self) {
        self.in_batch = true;
        self.pending.clear();
    }

    /// Add a transition to the current batch.
    ///
    /// Returns `true` if a barrier would be needed.
    pub fn batch_transition(
        &mut self,
        resource_id: u32,
        new_state: DemoResourceState,
    ) -> bool {
        if let Some(transition) = self.transition(resource_id, new_state) {
            if transition.requires_barrier() {
                self.pending.push(transition);
                return true;
            }
        }
        false
    }

    /// End the batch and return all pending transitions.
    ///
    /// If the batch contains transitions, this counts as one barrier.
    pub fn end_batch(&mut self) -> Vec<DemoResourceTransition> {
        self.in_batch = false;
        if !self.pending.is_empty() {
            self.barriers_this_frame += 1;
        }
        std::mem::take(&mut self.pending)
    }

    /// Reset frame statistics.
    pub fn begin_frame(&mut self) {
        self.barriers_this_frame = 0;
    }

    /// Get the number of barriers issued this frame.
    pub fn barriers_this_frame(&self) -> u32 {
        self.barriers_this_frame
    }

    /// Reset all resources to uninitialized state.
    pub fn reset(&mut self) {
        for (_, (_, state, _)) in self.states.iter_mut() {
            *state = DemoResourceState::Uninitialized;
        }
        self.pending.clear();
        self.barriers_this_frame = 0;
        self.in_batch = false;
    }
}

// ---------------------------------------------------------------------------
// Demoscene Frame Barrier Planner
// ---------------------------------------------------------------------------

/// Resource IDs for demoscene render targets.
pub mod resource_ids {
    pub const COLOR_BUFFER: u32 = 0;
    pub const DEPTH_BUFFER: u32 = 1;
    pub const NORMAL_BUFFER: u32 = 2;
    pub const MATERIAL_BUFFER: u32 = 3;
}

/// Pre-planned barrier sequence for demoscene rendering.
///
/// Implements the optimal 1-2 barrier strategy:
/// 1. **Pre-S13 barrier**: Transition depth buffer to compute write (if needed)
/// 2. **Post-S13 barrier**: Transition all outputs to rasterization inputs
#[derive(Debug)]
pub struct DemoFrameBarriers {
    scheduler: DemoBarrierScheduler,
    /// Transitions needed before S13 compute pass.
    pre_s13: Vec<DemoResourceTransition>,
    /// Transitions needed after S13 compute pass.
    post_s13: Vec<DemoResourceTransition>,
}

impl DemoFrameBarriers {
    /// Create a new frame barrier planner.
    pub fn new() -> Self {
        let mut scheduler = DemoBarrierScheduler::new();

        // Register standard demoscene resources
        scheduler.register_resource(
            resource_ids::COLOR_BUFFER,
            "color_buffer",
            DemoResourceState::Uninitialized,
            true,
        );
        scheduler.register_resource(
            resource_ids::DEPTH_BUFFER,
            "depth_buffer",
            DemoResourceState::Uninitialized,
            true,
        );
        scheduler.register_resource(
            resource_ids::NORMAL_BUFFER,
            "normal_buffer",
            DemoResourceState::Uninitialized,
            true,
        );
        scheduler.register_resource(
            resource_ids::MATERIAL_BUFFER,
            "material_buffer",
            DemoResourceState::Uninitialized,
            true,
        );

        Self {
            scheduler,
            pre_s13: Vec::new(),
            post_s13: Vec::new(),
        }
    }

    /// Plan barriers for a new frame.
    pub fn plan_frame(&mut self) {
        self.scheduler.begin_frame();
        self.pre_s13.clear();
        self.post_s13.clear();

        // Plan pre-S13 barriers (resources entering compute write state)
        self.scheduler.begin_batch();
        self.scheduler
            .batch_transition(resource_ids::COLOR_BUFFER, DemoResourceState::ComputeWrite);
        self.scheduler
            .batch_transition(resource_ids::DEPTH_BUFFER, DemoResourceState::ComputeWrite);
        self.scheduler
            .batch_transition(resource_ids::NORMAL_BUFFER, DemoResourceState::ComputeWrite);
        self.scheduler.batch_transition(
            resource_ids::MATERIAL_BUFFER,
            DemoResourceState::ComputeWrite,
        );
        self.pre_s13 = self.scheduler.end_batch();

        // Plan post-S13 barriers (resources entering rasterization state)
        self.scheduler.begin_batch();
        self.scheduler.batch_transition(
            resource_ids::COLOR_BUFFER,
            DemoResourceState::ColorAttachmentWrite,
        );
        self.scheduler.batch_transition(
            resource_ids::DEPTH_BUFFER,
            DemoResourceState::DepthAttachmentWrite,
        );
        self.scheduler
            .batch_transition(resource_ids::NORMAL_BUFFER, DemoResourceState::ShaderRead);
        self.scheduler
            .batch_transition(resource_ids::MATERIAL_BUFFER, DemoResourceState::ShaderRead);
        self.post_s13 = self.scheduler.end_batch();
    }

    /// Get transitions needed before S13 compute pass.
    pub fn pre_s13_barriers(&self) -> &[DemoResourceTransition] {
        &self.pre_s13
    }

    /// Get transitions needed after S13 compute pass.
    pub fn post_s13_barriers(&self) -> &[DemoResourceTransition] {
        &self.post_s13
    }

    /// Get total barriers for the frame.
    pub fn total_barriers(&self) -> u32 {
        self.scheduler.barriers_this_frame()
    }

    /// Reset for next frame.
    pub fn reset(&mut self) {
        self.scheduler.reset();
        self.pre_s13.clear();
        self.post_s13.clear();
    }
}

impl Default for DemoFrameBarriers {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WGSL Shader Include for Depth Reconstruction
// ---------------------------------------------------------------------------

/// WGSL shader code for depth reconstruction.
///
/// This can be concatenated with the main demoscene shader to add
/// depth buffer output capability.
pub const DEPTH_RECONSTRUCTION_WGSL: &str = r#"
// =============================================================================
// Depth Reconstruction (T-DEMO-6.5)
// =============================================================================

// Depth projection parameters: [near, far, near*far, far-near]
struct DepthParams {
    near: f32,
    far: f32,
    near_times_far: f32,
    far_minus_near: f32,
}

// Convert linear eye-space depth to NDC depth (reversed-Z).
fn linear_to_ndc_depth(z_linear: f32, params: DepthParams) -> f32 {
    if (z_linear <= 0.0) {
        return 1.0; // Near plane for reversed-Z
    }
    return clamp(params.near / z_linear, 0.0, 1.0);
}

// Convert NDC depth back to linear eye-space depth.
fn ndc_to_linear_depth(z_ndc: f32, params: DepthParams) -> f32 {
    if (z_ndc <= 0.0) {
        return params.far;
    }
    return clamp(params.near / z_ndc, params.near, params.far);
}

// Reconstruct depth from ray march hit distance.
// Returns: x = NDC depth, y = linear depth
fn reconstruct_depth(hit_distance: f32, params: DepthParams) -> vec2<f32> {
    let z_linear = hit_distance;
    let z_ndc = linear_to_ndc_depth(z_linear, params);
    return vec2<f32>(z_ndc, z_linear);
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // T-DEMO-6.5: Depth Reconstruction Tests
    // =========================================================================

    mod depth_projection_tests {
        use super::*;

        #[test]
        fn test_depth_projection_default() {
            let proj = DepthProjection::default();
            assert_eq!(proj.near, 0.1);
            assert_eq!(proj.far, 1000.0);
            assert!(proj.reversed_z);
        }

        #[test]
        fn test_depth_projection_new_clamps_values() {
            let proj = DepthProjection::new(-1.0, -5.0);
            assert!(proj.near > 0.0);
            assert!(proj.far > proj.near);
        }

        #[test]
        fn test_reversed_z_near_plane() {
            let proj = DepthProjection::reversed(0.1, 100.0);
            // At near plane, depth should be 1.0
            let ndc = proj.linear_to_ndc(0.1);
            assert!((ndc - 1.0).abs() < 0.001, "Near plane should map to 1.0, got {}", ndc);
        }

        #[test]
        fn test_reversed_z_far_plane() {
            let proj = DepthProjection::reversed(0.1, 100.0);
            // At far plane, depth should be near/far = 0.001
            let ndc = proj.linear_to_ndc(100.0);
            assert!(ndc < 0.01, "Far plane should map to near 0, got {}", ndc);
        }

        #[test]
        fn test_reversed_z_mid_range() {
            let proj = DepthProjection::reversed(0.1, 100.0);
            let ndc = proj.linear_to_ndc(10.0);
            // At z=10, ndc = 0.1/10 = 0.01
            assert!((ndc - 0.01).abs() < 0.001, "Mid-range depth incorrect, got {}", ndc);
        }

        #[test]
        fn test_standard_depth_near_plane() {
            let proj = DepthProjection::standard(0.1, 100.0);
            let ndc = proj.linear_to_ndc(0.1);
            assert!((ndc - 0.0).abs() < 0.001, "Near plane should map to 0.0, got {}", ndc);
        }

        #[test]
        fn test_standard_depth_far_plane() {
            let proj = DepthProjection::standard(0.1, 100.0);
            let ndc = proj.linear_to_ndc(100.0);
            assert!((ndc - 1.0).abs() < 0.001, "Far plane should map to 1.0, got {}", ndc);
        }

        #[test]
        fn test_round_trip_reversed_z() {
            let proj = DepthProjection::reversed(0.1, 1000.0);
            let test_depths = [0.1, 0.5, 1.0, 10.0, 100.0, 500.0, 1000.0];

            for &linear in &test_depths {
                let ndc = proj.linear_to_ndc(linear);
                let reconstructed = proj.ndc_to_linear(ndc);
                let error = (reconstructed - linear).abs() / linear;
                assert!(
                    error < 0.001,
                    "Round-trip failed for linear={}: ndc={}, reconstructed={}, error={}%",
                    linear, ndc, reconstructed, error * 100.0
                );
            }
        }

        #[test]
        fn test_round_trip_standard_depth() {
            let proj = DepthProjection::standard(0.1, 1000.0);
            let test_depths = [0.1, 0.5, 1.0, 10.0, 100.0, 500.0, 1000.0];

            for &linear in &test_depths {
                let ndc = proj.linear_to_ndc(linear);
                let reconstructed = proj.ndc_to_linear(ndc);
                let error = (reconstructed - linear).abs() / linear;
                assert!(
                    error < 0.01,
                    "Round-trip failed for linear={}: ndc={}, reconstructed={}, error={}%",
                    linear, ndc, reconstructed, error * 100.0
                );
            }
        }

        #[test]
        fn test_behind_camera_reversed() {
            let proj = DepthProjection::reversed(0.1, 100.0);
            let ndc = proj.linear_to_ndc(-5.0);
            assert_eq!(ndc, 1.0, "Behind camera should return near plane value (1.0)");
        }

        #[test]
        fn test_behind_camera_standard() {
            let proj = DepthProjection::standard(0.1, 100.0);
            let ndc = proj.linear_to_ndc(-5.0);
            assert_eq!(ndc, 0.0, "Behind camera should return near plane value (0.0)");
        }

        #[test]
        fn test_gpu_params() {
            let proj = DepthProjection::new(0.1, 100.0);
            let params = proj.to_gpu_params();
            assert_eq!(params[0], 0.1);
            assert_eq!(params[1], 100.0);
            assert!((params[2] - 10.0).abs() < 0.001); // near * far
            assert!((params[3] - 99.9).abs() < 0.001); // far - near
        }

        #[test]
        fn test_linearize_depth() {
            let proj = DepthProjection::reversed(0.1, 100.0);
            let ndc = 0.01; // Corresponds to z=10 in reversed-Z
            let linear = proj.linearize_depth(ndc);
            assert!((linear - 10.0).abs() < 0.1);
        }
    }

    // =========================================================================
    // Depth Reconstruction Result Tests
    // =========================================================================

    mod reconstruction_result_tests {
        use super::*;

        #[test]
        fn test_miss_result() {
            let result = DepthReconstructionResult::miss();
            assert!(!result.hit);
            assert!(result.linear_depth.is_infinite());
            assert_eq!(result.ndc_depth, 0.0); // Far plane in reversed-Z
        }

        #[test]
        fn test_hit_result() {
            let proj = DepthProjection::reversed(0.1, 100.0);
            let result = DepthReconstructionResult::hit(10.0, &proj, 42);
            assert!(result.hit);
            assert_eq!(result.linear_depth, 10.0);
            assert!((result.ndc_depth - 0.01).abs() < 0.001);
            assert_eq!(result.material_id, 42);
        }

        #[test]
        fn test_from_ndc() {
            let proj = DepthProjection::reversed(0.1, 100.0);
            let result = DepthReconstructionResult::from_ndc(0.01, &proj);
            assert!(result.hit);
            assert!((result.linear_depth - 10.0).abs() < 0.1);
        }
    }

    // =========================================================================
    // T-DEMO-6.6: Resource Transition Tests
    // =========================================================================

    mod resource_state_tests {
        use super::*;

        #[test]
        fn test_read_only_states() {
            assert!(DemoResourceState::ComputeRead.is_read_only());
            assert!(DemoResourceState::ShaderRead.is_read_only());
            assert!(DemoResourceState::DepthAttachmentReadOnly.is_read_only());
            assert!(DemoResourceState::TransferSrc.is_read_only());
        }

        #[test]
        fn test_write_states() {
            assert!(DemoResourceState::ComputeWrite.is_write());
            assert!(DemoResourceState::ColorAttachmentWrite.is_write());
            assert!(DemoResourceState::DepthAttachmentWrite.is_write());
            assert!(DemoResourceState::TransferDst.is_write());
        }

        #[test]
        fn test_same_state_no_barrier() {
            let state = DemoResourceState::ComputeWrite;
            assert!(!state.needs_barrier_to(state));
        }

        #[test]
        fn test_write_to_read_needs_barrier() {
            assert!(DemoResourceState::ComputeWrite.needs_barrier_to(DemoResourceState::ShaderRead));
        }

        #[test]
        fn test_read_to_write_needs_barrier() {
            assert!(DemoResourceState::ShaderRead.needs_barrier_to(DemoResourceState::ComputeWrite));
        }
    }

    // =========================================================================
    // Barrier Scheduler Tests
    // =========================================================================

    mod scheduler_tests {
        use super::*;

        #[test]
        fn test_register_resource() {
            let mut scheduler = DemoBarrierScheduler::new();
            scheduler.register_resource(1, "test", DemoResourceState::Uninitialized, true);
            assert_eq!(
                scheduler.get_state(1),
                Some(DemoResourceState::Uninitialized)
            );
        }

        #[test]
        fn test_transition_generates_barrier() {
            let mut scheduler = DemoBarrierScheduler::new();
            scheduler.register_resource(1, "color", DemoResourceState::ComputeWrite, true);

            let result = scheduler.transition(1, DemoResourceState::ShaderRead);
            assert!(result.is_some());
            let transition = result.unwrap();
            assert_eq!(transition.before, DemoResourceState::ComputeWrite);
            assert_eq!(transition.after, DemoResourceState::ShaderRead);
        }

        #[test]
        fn test_same_state_no_transition() {
            let mut scheduler = DemoBarrierScheduler::new();
            scheduler.register_resource(1, "test", DemoResourceState::ComputeWrite, true);

            let result = scheduler.transition(1, DemoResourceState::ComputeWrite);
            assert!(result.is_none());
        }

        #[test]
        fn test_batch_transitions() {
            let mut scheduler = DemoBarrierScheduler::new();
            scheduler.register_resource(1, "color", DemoResourceState::ComputeWrite, true);
            scheduler.register_resource(2, "depth", DemoResourceState::ComputeWrite, true);

            scheduler.begin_batch();
            scheduler.batch_transition(1, DemoResourceState::ShaderRead);
            scheduler.batch_transition(2, DemoResourceState::DepthAttachmentReadOnly);
            let transitions = scheduler.end_batch();

            assert_eq!(transitions.len(), 2);
            assert_eq!(scheduler.barriers_this_frame(), 1); // Single batched barrier
        }

        #[test]
        fn test_empty_batch() {
            let mut scheduler = DemoBarrierScheduler::new();
            scheduler.register_resource(1, "test", DemoResourceState::ComputeWrite, true);

            scheduler.begin_batch();
            // No transitions added
            let transitions = scheduler.end_batch();

            assert!(transitions.is_empty());
            assert_eq!(scheduler.barriers_this_frame(), 0);
        }

        #[test]
        fn test_reset_frame_stats() {
            let mut scheduler = DemoBarrierScheduler::new();
            scheduler.register_resource(1, "test", DemoResourceState::Uninitialized, true);

            scheduler.transition(1, DemoResourceState::ComputeWrite);
            scheduler.begin_frame();

            assert_eq!(scheduler.barriers_this_frame(), 0);
        }

        #[test]
        fn test_unknown_resource() {
            let scheduler = DemoBarrierScheduler::new();
            assert_eq!(scheduler.get_state(999), None);
        }
    }

    // =========================================================================
    // Demo Frame Barriers Tests
    // =========================================================================

    mod frame_barriers_tests {
        use super::*;

        #[test]
        fn test_frame_barriers_plan() {
            let mut barriers = DemoFrameBarriers::new();
            barriers.plan_frame();

            // Should have transitions for S13 outputs
            assert!(!barriers.pre_s13_barriers().is_empty());
            assert!(!barriers.post_s13_barriers().is_empty());
        }

        #[test]
        fn test_frame_barriers_max_two() {
            let mut barriers = DemoFrameBarriers::new();
            barriers.plan_frame();

            // Target: 1-2 barriers per frame
            assert!(
                barriers.total_barriers() <= 2,
                "Expected at most 2 barriers, got {}",
                barriers.total_barriers()
            );
        }

        #[test]
        fn test_frame_barriers_reset() {
            let mut barriers = DemoFrameBarriers::new();
            barriers.plan_frame();
            barriers.reset();

            assert!(barriers.pre_s13_barriers().is_empty());
            assert!(barriers.post_s13_barriers().is_empty());
        }

        #[test]
        fn test_multiple_frames() {
            let mut barriers = DemoFrameBarriers::new();

            for _ in 0..10 {
                barriers.plan_frame();
                assert!(barriers.total_barriers() <= 2);
            }
        }
    }

    // =========================================================================
    // Resource Transition Tests
    // =========================================================================

    mod transition_tests {
        use super::*;

        #[test]
        fn test_transition_display() {
            let transition = DemoResourceTransition::new(
                1,
                "color_buffer",
                DemoResourceState::ComputeWrite,
                DemoResourceState::ShaderRead,
                true,
            );

            let display = format!("{}", transition);
            assert!(display.contains("color_buffer"));
            assert!(display.contains("ComputeWrite"));
            assert!(display.contains("ShaderRead"));
        }

        #[test]
        fn test_transition_requires_barrier() {
            let transition = DemoResourceTransition::new(
                1,
                "test",
                DemoResourceState::ComputeWrite,
                DemoResourceState::ShaderRead,
                true,
            );
            assert!(transition.requires_barrier());
        }

        #[test]
        fn test_same_state_no_barrier_required() {
            let transition = DemoResourceTransition::new(
                1,
                "test",
                DemoResourceState::ComputeWrite,
                DemoResourceState::ComputeWrite,
                true,
            );
            assert!(!transition.requires_barrier());
        }
    }

    // =========================================================================
    // WGSL Shader Include Tests
    // =========================================================================

    #[test]
    fn test_wgsl_shader_include_not_empty() {
        assert!(!DEPTH_RECONSTRUCTION_WGSL.is_empty());
    }

    #[test]
    fn test_wgsl_shader_contains_depth_functions() {
        assert!(DEPTH_RECONSTRUCTION_WGSL.contains("linear_to_ndc_depth"));
        assert!(DEPTH_RECONSTRUCTION_WGSL.contains("ndc_to_linear_depth"));
        assert!(DEPTH_RECONSTRUCTION_WGSL.contains("reconstruct_depth"));
    }

    #[test]
    fn test_wgsl_shader_contains_struct() {
        assert!(DEPTH_RECONSTRUCTION_WGSL.contains("struct DepthParams"));
    }

    // =========================================================================
    // Accuracy Tests (50+ tests target)
    // =========================================================================

    mod accuracy_tests {
        use super::*;

        #[test]
        fn test_depth_conversion_accuracy_01() {
            let proj = DepthProjection::reversed(0.01, 10000.0);
            assert_depth_accuracy(&proj, 0.01, 0.001);
        }

        #[test]
        fn test_depth_conversion_accuracy_02() {
            let proj = DepthProjection::reversed(0.01, 10000.0);
            assert_depth_accuracy(&proj, 0.1, 0.001);
        }

        #[test]
        fn test_depth_conversion_accuracy_03() {
            let proj = DepthProjection::reversed(0.01, 10000.0);
            assert_depth_accuracy(&proj, 1.0, 0.001);
        }

        #[test]
        fn test_depth_conversion_accuracy_04() {
            let proj = DepthProjection::reversed(0.01, 10000.0);
            assert_depth_accuracy(&proj, 10.0, 0.001);
        }

        #[test]
        fn test_depth_conversion_accuracy_05() {
            let proj = DepthProjection::reversed(0.01, 10000.0);
            assert_depth_accuracy(&proj, 100.0, 0.001);
        }

        #[test]
        fn test_depth_conversion_accuracy_06() {
            let proj = DepthProjection::reversed(0.01, 10000.0);
            assert_depth_accuracy(&proj, 1000.0, 0.001);
        }

        #[test]
        fn test_depth_conversion_accuracy_07() {
            let proj = DepthProjection::reversed(0.01, 10000.0);
            assert_depth_accuracy(&proj, 5000.0, 0.001);
        }

        #[test]
        fn test_depth_conversion_accuracy_08() {
            let proj = DepthProjection::reversed(0.01, 10000.0);
            assert_depth_accuracy(&proj, 10000.0, 0.001);
        }

        #[test]
        fn test_depth_conversion_accuracy_09() {
            let proj = DepthProjection::standard(0.1, 1000.0);
            assert_depth_accuracy(&proj, 0.1, 0.01);
        }

        #[test]
        fn test_depth_conversion_accuracy_10() {
            let proj = DepthProjection::standard(0.1, 1000.0);
            assert_depth_accuracy(&proj, 10.0, 0.01);
        }

        fn assert_depth_accuracy(proj: &DepthProjection, linear: f32, tolerance: f32) {
            let ndc = proj.linear_to_ndc(linear);
            let reconstructed = proj.ndc_to_linear(ndc);
            let error = (reconstructed - linear).abs() / linear;
            assert!(
                error < tolerance,
                "Accuracy test failed: linear={}, ndc={}, reconstructed={}, error={}%",
                linear, ndc, reconstructed, error * 100.0
            );
        }
    }

    // =========================================================================
    // Barrier Count Performance Tests
    // =========================================================================

    mod barrier_count_tests {
        use super::*;

        #[test]
        fn test_single_frame_barrier_count() {
            let mut barriers = DemoFrameBarriers::new();
            barriers.plan_frame();
            assert!(barriers.total_barriers() <= 2);
        }

        #[test]
        fn test_hundred_frames_barrier_count() {
            let mut barriers = DemoFrameBarriers::new();
            let mut max_barriers = 0;

            for _ in 0..100 {
                barriers.plan_frame();
                max_barriers = max_barriers.max(barriers.total_barriers());
            }

            assert!(max_barriers <= 2, "Max barriers should be <= 2, got {}", max_barriers);
        }

        #[test]
        fn test_barrier_batching_efficiency() {
            let mut scheduler = DemoBarrierScheduler::new();
            scheduler.register_resource(1, "r1", DemoResourceState::ComputeWrite, true);
            scheduler.register_resource(2, "r2", DemoResourceState::ComputeWrite, true);
            scheduler.register_resource(3, "r3", DemoResourceState::ComputeWrite, true);
            scheduler.register_resource(4, "r4", DemoResourceState::ComputeWrite, true);

            scheduler.begin_batch();
            scheduler.batch_transition(1, DemoResourceState::ShaderRead);
            scheduler.batch_transition(2, DemoResourceState::ShaderRead);
            scheduler.batch_transition(3, DemoResourceState::ShaderRead);
            scheduler.batch_transition(4, DemoResourceState::ShaderRead);
            let transitions = scheduler.end_batch();

            assert_eq!(transitions.len(), 4);
            assert_eq!(scheduler.barriers_this_frame(), 1); // All batched into one
        }
    }
}
