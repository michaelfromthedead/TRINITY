//! Predictive Asset Pre-loading System
//!
//! This module implements predictive asset pre-loading based on camera movement
//! to minimize loading stalls and improve streaming performance.
//!
//! # Features
//!
//! - Camera velocity tracking with smoothing
//! - Frustum prediction for N frames ahead
//! - Asset scoring based on predicted visibility
//! - Priority queue integration with rate limiting
//! - Request cancellation for no-longer-needed assets
//!
//! # Algorithm
//!
//! 1. Track camera position and velocity over time
//! 2. Predict camera position in N frames using velocity extrapolation
//! 3. For each potential asset, compute weighted score:
//!    `score = visibility * vis_w + dot(to_asset, velocity) * vel_w + (1/distance) * dist_w + lod_bias * bias_w`
//! 4. Sort by score, submit top K to preload queue
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::preload::{PreloadManager, PreloadWeights, CameraState};
//!
//! let mut manager = PreloadManager::new(PreloadWeights::default());
//!
//! // Update camera each frame
//! manager.update_camera(CameraState {
//!     position: [0.0, 0.0, 0.0],
//!     direction: [0.0, 0.0, -1.0],
//!     fov: 60.0,
//! }, 0.016);
//!
//! // Compute predictions and submit requests
//! let requests = manager.predict_and_queue(&assets, 10);
//! ```

use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// ContentHash
// ---------------------------------------------------------------------------

/// Content-addressable hash for assets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub u64);

impl ContentHash {
    /// Create a new content hash from raw value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Get the raw hash value.
    pub const fn value(&self) -> u64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// PreloadWeights
// ---------------------------------------------------------------------------

/// Weights for computing preload priority scores.
///
/// Each weight controls how much influence that factor has on the final
/// preload priority. Weights should sum to approximately 1.0 for normalized
/// scores, but this is not enforced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreloadWeights {
    /// Weight for visibility state (0 = invisible, 1 = visible).
    pub visibility: f32,
    /// Weight for camera velocity direction alignment.
    pub velocity: f32,
    /// Weight for inverse distance to camera.
    pub distance: f32,
    /// Weight for LOD bias preference.
    pub lod_bias: f32,
}

impl Default for PreloadWeights {
    fn default() -> Self {
        Self {
            visibility: 0.3,
            velocity: 0.3,
            distance: 0.25,
            lod_bias: 0.15,
        }
    }
}

impl PreloadWeights {
    /// Create custom weights.
    pub const fn new(visibility: f32, velocity: f32, distance: f32, lod_bias: f32) -> Self {
        Self {
            visibility,
            velocity,
            distance,
            lod_bias,
        }
    }

    /// Sum of all weights.
    pub fn total(&self) -> f32 {
        self.visibility + self.velocity + self.distance + self.lod_bias
    }

    /// Create normalized weights that sum to 1.0.
    pub fn normalized(&self) -> Self {
        let total = self.total();
        if total <= 0.0 {
            return Self::default();
        }
        Self {
            visibility: self.visibility / total,
            velocity: self.velocity / total,
            distance: self.distance / total,
            lod_bias: self.lod_bias / total,
        }
    }
}

// ---------------------------------------------------------------------------
// CameraState
// ---------------------------------------------------------------------------

/// Camera state for a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraState {
    /// World-space position [x, y, z].
    pub position: [f32; 3],
    /// Normalized view direction [x, y, z].
    pub direction: [f32; 3],
    /// Vertical field of view in degrees.
    pub fov: f32,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, -1.0],
            fov: 60.0,
        }
    }
}

impl CameraState {
    /// Create a new camera state.
    pub const fn new(position: [f32; 3], direction: [f32; 3], fov: f32) -> Self {
        Self {
            position,
            direction,
            fov,
        }
    }
}

// ---------------------------------------------------------------------------
// CameraVelocityTracker
// ---------------------------------------------------------------------------

/// Tracks camera velocity over time with exponential smoothing.
#[derive(Debug, Clone)]
pub struct CameraVelocityTracker {
    /// Previous camera position.
    prev_position: [f32; 3],
    /// Smoothed velocity vector.
    velocity: [f32; 3],
    /// Smoothing factor (0 = no smoothing, 1 = instant).
    smoothing: f32,
    /// Whether we have a valid previous sample.
    initialized: bool,
}

impl Default for CameraVelocityTracker {
    fn default() -> Self {
        Self::new(0.2)
    }
}

impl CameraVelocityTracker {
    /// Create a new velocity tracker with the given smoothing factor.
    ///
    /// # Parameters
    ///
    /// * `smoothing` - Exponential smoothing factor in range [0, 1].
    ///   Lower values produce smoother but more latent velocity estimates.
    pub fn new(smoothing: f32) -> Self {
        Self {
            prev_position: [0.0; 3],
            velocity: [0.0; 3],
            smoothing: smoothing.clamp(0.0, 1.0),
            initialized: false,
        }
    }

    /// Update the tracker with a new camera position.
    ///
    /// Returns the current smoothed velocity.
    pub fn update(&mut self, position: [f32; 3], delta_time: f32) -> [f32; 3] {
        if !self.initialized {
            self.prev_position = position;
            self.initialized = true;
            return self.velocity;
        }

        if delta_time <= 0.0 {
            return self.velocity;
        }

        // Compute instantaneous velocity
        let instant_vel = [
            (position[0] - self.prev_position[0]) / delta_time,
            (position[1] - self.prev_position[1]) / delta_time,
            (position[2] - self.prev_position[2]) / delta_time,
        ];

        // Exponential smoothing
        self.velocity = [
            self.velocity[0] + self.smoothing * (instant_vel[0] - self.velocity[0]),
            self.velocity[1] + self.smoothing * (instant_vel[1] - self.velocity[1]),
            self.velocity[2] + self.smoothing * (instant_vel[2] - self.velocity[2]),
        ];

        self.prev_position = position;
        self.velocity
    }

    /// Get the current smoothed velocity.
    pub fn velocity(&self) -> [f32; 3] {
        self.velocity
    }

    /// Get the velocity magnitude.
    pub fn speed(&self) -> f32 {
        let v = self.velocity;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// Reset the tracker state.
    pub fn reset(&mut self) {
        self.prev_position = [0.0; 3];
        self.velocity = [0.0; 3];
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// FrustumPredictor
// ---------------------------------------------------------------------------

/// Predicts future camera frustum based on velocity extrapolation.
#[derive(Debug, Clone)]
pub struct FrustumPredictor {
    /// Number of frames to predict ahead.
    prediction_frames: u32,
    /// Assumed frame time for prediction.
    frame_time: f32,
}

impl Default for FrustumPredictor {
    fn default() -> Self {
        Self::new(10, 1.0 / 60.0)
    }
}

impl FrustumPredictor {
    /// Create a new frustum predictor.
    ///
    /// # Parameters
    ///
    /// * `prediction_frames` - How many frames ahead to predict.
    /// * `frame_time` - Assumed time per frame in seconds.
    pub fn new(prediction_frames: u32, frame_time: f32) -> Self {
        Self {
            prediction_frames,
            frame_time,
        }
    }

    /// Predict the camera position in N frames.
    pub fn predict_position(&self, current: &CameraState, velocity: [f32; 3]) -> [f32; 3] {
        let time_ahead = self.prediction_frames as f32 * self.frame_time;
        [
            current.position[0] + velocity[0] * time_ahead,
            current.position[1] + velocity[1] * time_ahead,
            current.position[2] + velocity[2] * time_ahead,
        ]
    }

    /// Get the prediction time horizon in seconds.
    pub fn prediction_time(&self) -> f32 {
        self.prediction_frames as f32 * self.frame_time
    }

    /// Set the number of frames to predict ahead.
    pub fn set_prediction_frames(&mut self, frames: u32) {
        self.prediction_frames = frames;
    }

    /// Set the assumed frame time.
    pub fn set_frame_time(&mut self, time: f32) {
        self.frame_time = time.max(0.001);
    }
}

// ---------------------------------------------------------------------------
// AssetInfo
// ---------------------------------------------------------------------------

/// Information about an asset for preload scoring.
#[derive(Debug, Clone)]
pub struct AssetInfo {
    /// Unique content hash.
    pub hash: ContentHash,
    /// World-space position (center of bounds).
    pub position: [f32; 3],
    /// Bounding radius.
    pub radius: f32,
    /// LOD bias for this asset (negative = prefer higher detail).
    pub lod_bias: f32,
    /// Whether the asset is currently visible.
    pub visible: bool,
    /// Estimated load time in frames.
    pub load_frames: f32,
}

impl AssetInfo {
    /// Create a new asset info.
    pub fn new(hash: ContentHash, position: [f32; 3], radius: f32) -> Self {
        Self {
            hash,
            position,
            radius,
            lod_bias: 0.0,
            visible: false,
            load_frames: 1.0,
        }
    }

    /// Set the LOD bias.
    pub fn with_lod_bias(mut self, bias: f32) -> Self {
        self.lod_bias = bias;
        self
    }

    /// Set the visibility state.
    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Set the estimated load time.
    pub fn with_load_frames(mut self, frames: f32) -> Self {
        self.load_frames = frames;
        self
    }
}

// ---------------------------------------------------------------------------
// PreloadRequest
// ---------------------------------------------------------------------------

/// A request to preload an asset.
#[derive(Debug, Clone, PartialEq)]
pub struct PreloadRequest {
    /// Content hash of the asset to preload.
    pub asset_id: ContentHash,
    /// Computed priority score (higher = more urgent).
    pub priority: f32,
    /// Predicted frames until the asset is needed.
    pub predicted_need_time: f32,
}

impl PreloadRequest {
    /// Create a new preload request.
    pub fn new(asset_id: ContentHash, priority: f32, predicted_need_time: f32) -> Self {
        Self {
            asset_id,
            priority,
            predicted_need_time,
        }
    }
}

impl Eq for PreloadRequest {}

impl Ord for PreloadRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then sooner need time
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                other.predicted_need_time
                    .partial_cmp(&self.predicted_need_time)
                    .unwrap_or(Ordering::Equal)
            })
    }
}

impl PartialOrd for PreloadRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// PreloadQueue
// ---------------------------------------------------------------------------

/// Priority queue for preload requests with rate limiting.
#[derive(Debug)]
pub struct PreloadQueue {
    /// Priority heap of pending requests.
    heap: BinaryHeap<PreloadRequest>,
    /// Set of asset IDs currently in the queue.
    pending: HashSet<ContentHash>,
    /// Maximum requests to process per frame.
    rate_limit: usize,
    /// Currently active requests (being loaded).
    active: HashSet<ContentHash>,
    /// Maximum concurrent active requests.
    max_active: usize,
}

impl Default for PreloadQueue {
    fn default() -> Self {
        Self::new(5, 10)
    }
}

impl PreloadQueue {
    /// Create a new preload queue.
    ///
    /// # Parameters
    ///
    /// * `rate_limit` - Maximum requests to submit per frame.
    /// * `max_active` - Maximum concurrent active requests.
    pub fn new(rate_limit: usize, max_active: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            pending: HashSet::new(),
            rate_limit: rate_limit.max(1),
            active: HashSet::new(),
            max_active: max_active.max(1),
        }
    }

    /// Submit a preload request.
    ///
    /// Returns `true` if the request was added, `false` if already pending or active.
    pub fn submit(&mut self, request: PreloadRequest) -> bool {
        let id = request.asset_id;
        if self.pending.contains(&id) || self.active.contains(&id) {
            return false;
        }
        self.pending.insert(id);
        self.heap.push(request);
        true
    }

    /// Cancel a pending request.
    ///
    /// Note: This only removes from the pending set; the heap entry remains
    /// but will be skipped when dequeued.
    pub fn cancel(&mut self, asset_id: ContentHash) -> bool {
        self.pending.remove(&asset_id)
    }

    /// Cancel all pending requests.
    pub fn cancel_all(&mut self) {
        self.pending.clear();
        self.heap.clear();
    }

    /// Dequeue up to `rate_limit` requests for processing.
    ///
    /// Respects the `max_active` limit on concurrent requests.
    pub fn dequeue(&mut self) -> Vec<PreloadRequest> {
        let mut result = Vec::new();
        let available_slots = self.max_active.saturating_sub(self.active.len());
        let limit = self.rate_limit.min(available_slots);

        while result.len() < limit {
            match self.heap.pop() {
                Some(req) => {
                    if self.pending.remove(&req.asset_id) {
                        self.active.insert(req.asset_id);
                        result.push(req);
                    }
                    // Skip if not in pending (was cancelled)
                }
                None => break,
            }
        }

        result
    }

    /// Mark a request as completed (no longer active).
    pub fn complete(&mut self, asset_id: ContentHash) {
        self.active.remove(&asset_id);
    }

    /// Get the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get the number of active requests.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Check if an asset is pending or active.
    pub fn contains(&self, asset_id: ContentHash) -> bool {
        self.pending.contains(&asset_id) || self.active.contains(&asset_id)
    }

    /// Get the rate limit.
    pub fn rate_limit(&self) -> usize {
        self.rate_limit
    }

    /// Set the rate limit.
    pub fn set_rate_limit(&mut self, limit: usize) {
        self.rate_limit = limit.max(1);
    }
}

// ---------------------------------------------------------------------------
// AssetScorer
// ---------------------------------------------------------------------------

/// Computes preload priority scores for assets.
#[derive(Debug, Clone)]
pub struct AssetScorer {
    /// Scoring weights.
    weights: PreloadWeights,
    /// Maximum distance for scoring (assets beyond this get score 0).
    max_distance: f32,
}

impl Default for AssetScorer {
    fn default() -> Self {
        Self::new(PreloadWeights::default(), 1000.0)
    }
}

impl AssetScorer {
    /// Create a new asset scorer.
    pub fn new(weights: PreloadWeights, max_distance: f32) -> Self {
        Self {
            weights,
            max_distance: max_distance.max(1.0),
        }
    }

    /// Compute the preload priority score for an asset.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// score = visibility * vis_w
    ///       + dot(to_asset, velocity) * vel_w
    ///       + (1 / distance) * dist_w
    ///       + lod_bias * bias_w
    /// ```
    pub fn score(
        &self,
        asset: &AssetInfo,
        camera_pos: [f32; 3],
        velocity: [f32; 3],
    ) -> f32 {
        // Compute direction to asset
        let to_asset = [
            asset.position[0] - camera_pos[0],
            asset.position[1] - camera_pos[1],
            asset.position[2] - camera_pos[2],
        ];

        // Distance to asset
        let distance = (to_asset[0] * to_asset[0]
            + to_asset[1] * to_asset[1]
            + to_asset[2] * to_asset[2])
        .sqrt()
        .max(0.001);

        if distance > self.max_distance {
            return 0.0;
        }

        // Normalize to_asset direction
        let to_asset_norm = [
            to_asset[0] / distance,
            to_asset[1] / distance,
            to_asset[2] / distance,
        ];

        // Normalize velocity (or use zero if stationary)
        let vel_mag = (velocity[0] * velocity[0]
            + velocity[1] * velocity[1]
            + velocity[2] * velocity[2])
        .sqrt();
        let vel_norm = if vel_mag > 0.001 {
            [
                velocity[0] / vel_mag,
                velocity[1] / vel_mag,
                velocity[2] / vel_mag,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };

        // Dot product: alignment between camera velocity and direction to asset
        let velocity_alignment =
            to_asset_norm[0] * vel_norm[0]
            + to_asset_norm[1] * vel_norm[1]
            + to_asset_norm[2] * vel_norm[2];

        // Clamp alignment to [0, 1] (only care about moving towards asset)
        let velocity_score = velocity_alignment.max(0.0);

        // Visibility score
        let visibility_score = if asset.visible { 1.0 } else { 0.0 };

        // Distance score (inverse, normalized to max distance)
        let distance_score = 1.0 - (distance / self.max_distance).clamp(0.0, 1.0);

        // LOD bias score (normalize to [-1, 1] range, then shift to [0, 1])
        let lod_score = (asset.lod_bias.clamp(-1.0, 1.0) + 1.0) / 2.0;

        // Weighted sum
        self.weights.visibility * visibility_score
            + self.weights.velocity * velocity_score
            + self.weights.distance * distance_score
            + self.weights.lod_bias * lod_score
    }

    /// Get the weights.
    pub fn weights(&self) -> &PreloadWeights {
        &self.weights
    }

    /// Set the weights.
    pub fn set_weights(&mut self, weights: PreloadWeights) {
        self.weights = weights;
    }
}

// ---------------------------------------------------------------------------
// PreloadManager
// ---------------------------------------------------------------------------

/// Main interface for the predictive preloading system.
#[derive(Debug)]
pub struct PreloadManager {
    /// Camera velocity tracker.
    velocity_tracker: CameraVelocityTracker,
    /// Frustum predictor.
    predictor: FrustumPredictor,
    /// Asset scorer.
    scorer: AssetScorer,
    /// Preload queue.
    queue: PreloadQueue,
    /// Current camera state.
    current_camera: CameraState,
}

impl Default for PreloadManager {
    fn default() -> Self {
        Self::new(PreloadWeights::default())
    }
}

impl PreloadManager {
    /// Create a new preload manager with the given weights.
    pub fn new(weights: PreloadWeights) -> Self {
        Self {
            velocity_tracker: CameraVelocityTracker::default(),
            predictor: FrustumPredictor::default(),
            scorer: AssetScorer::new(weights, 1000.0),
            queue: PreloadQueue::default(),
            current_camera: CameraState::default(),
        }
    }

    /// Create a preload manager with custom configuration.
    pub fn with_config(
        weights: PreloadWeights,
        prediction_frames: u32,
        rate_limit: usize,
        max_active: usize,
    ) -> Self {
        Self {
            velocity_tracker: CameraVelocityTracker::default(),
            predictor: FrustumPredictor::new(prediction_frames, 1.0 / 60.0),
            scorer: AssetScorer::new(weights, 1000.0),
            queue: PreloadQueue::new(rate_limit, max_active),
            current_camera: CameraState::default(),
        }
    }

    /// Update the camera state and velocity tracking.
    pub fn update_camera(&mut self, camera: CameraState, delta_time: f32) {
        self.velocity_tracker.update(camera.position, delta_time);
        self.current_camera = camera;
    }

    /// Get the current smoothed camera velocity.
    pub fn camera_velocity(&self) -> [f32; 3] {
        self.velocity_tracker.velocity()
    }

    /// Get the predicted camera position.
    pub fn predicted_position(&self) -> [f32; 3] {
        self.predictor
            .predict_position(&self.current_camera, self.velocity_tracker.velocity())
    }

    /// Score and queue assets for preloading.
    ///
    /// # Parameters
    ///
    /// * `assets` - Slice of assets to consider.
    /// * `top_k` - Maximum number of assets to queue.
    ///
    /// # Returns
    ///
    /// Number of requests submitted to the queue.
    pub fn predict_and_queue(&mut self, assets: &[AssetInfo], top_k: usize) -> usize {
        if assets.is_empty() || top_k == 0 {
            return 0;
        }

        let velocity = self.velocity_tracker.velocity();
        let predicted_pos = self.predictor.predict_position(&self.current_camera, velocity);

        // Score all assets
        let mut scored: Vec<(f32, &AssetInfo)> = assets
            .iter()
            .map(|asset| (self.scorer.score(asset, predicted_pos, velocity), asset))
            .filter(|(score, _)| *score > 0.0)
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Submit top K
        let mut submitted = 0;
        for (score, asset) in scored.into_iter().take(top_k) {
            let request = PreloadRequest::new(
                asset.hash,
                score,
                asset.load_frames,
            );
            if self.queue.submit(request) {
                submitted += 1;
            }
        }

        submitted
    }

    /// Dequeue ready requests for processing.
    pub fn dequeue(&mut self) -> Vec<PreloadRequest> {
        self.queue.dequeue()
    }

    /// Mark a preload as completed.
    pub fn complete(&mut self, asset_id: ContentHash) {
        self.queue.complete(asset_id);
    }

    /// Cancel a pending preload.
    pub fn cancel(&mut self, asset_id: ContentHash) -> bool {
        self.queue.cancel(asset_id)
    }

    /// Cancel all pending preloads.
    pub fn cancel_all(&mut self) {
        self.queue.cancel_all();
    }

    /// Get the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.queue.pending_count()
    }

    /// Get the number of active requests.
    pub fn active_count(&self) -> usize {
        self.queue.active_count()
    }

    /// Access the preload queue directly.
    pub fn queue(&self) -> &PreloadQueue {
        &self.queue
    }

    /// Access the preload queue mutably.
    pub fn queue_mut(&mut self) -> &mut PreloadQueue {
        &mut self.queue
    }

    /// Access the scorer.
    pub fn scorer(&self) -> &AssetScorer {
        &self.scorer
    }

    /// Access the scorer mutably.
    pub fn scorer_mut(&mut self) -> &mut AssetScorer {
        &mut self.scorer
    }

    /// Access the predictor.
    pub fn predictor(&self) -> &FrustumPredictor {
        &self.predictor
    }

    /// Access the predictor mutably.
    pub fn predictor_mut(&mut self) -> &mut FrustumPredictor {
        &mut self.predictor
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.velocity_tracker.reset();
        self.queue.cancel_all();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ContentHash ----

    #[test]
    fn test_content_hash_creation() {
        let hash = ContentHash::new(12345);
        assert_eq!(hash.value(), 12345);
    }

    #[test]
    fn test_content_hash_equality() {
        let a = ContentHash::new(100);
        let b = ContentHash::new(100);
        let c = ContentHash::new(200);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ---- PreloadWeights ----

    #[test]
    fn test_weights_default() {
        let w = PreloadWeights::default();
        assert_eq!(w.visibility, 0.3);
        assert_eq!(w.velocity, 0.3);
        assert_eq!(w.distance, 0.25);
        assert_eq!(w.lod_bias, 0.15);
    }

    #[test]
    fn test_weights_total() {
        let w = PreloadWeights::default();
        let total = w.total();
        assert!((total - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weights_normalized() {
        let w = PreloadWeights::new(0.6, 0.6, 0.5, 0.3);
        let n = w.normalized();
        assert!((n.total() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weights_normalized_zero_total() {
        let w = PreloadWeights::new(0.0, 0.0, 0.0, 0.0);
        let n = w.normalized();
        assert_eq!(n, PreloadWeights::default());
    }

    // ---- CameraVelocityTracker ----

    #[test]
    fn test_velocity_tracker_initial() {
        let tracker = CameraVelocityTracker::default();
        assert_eq!(tracker.velocity(), [0.0, 0.0, 0.0]);
        assert_eq!(tracker.speed(), 0.0);
    }

    #[test]
    fn test_velocity_tracker_first_update() {
        let mut tracker = CameraVelocityTracker::new(1.0);
        tracker.update([10.0, 0.0, 0.0], 0.016);
        // First update only stores position, returns zero velocity
        assert_eq!(tracker.velocity(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_velocity_tracker_second_update() {
        let mut tracker = CameraVelocityTracker::new(1.0);
        tracker.update([0.0, 0.0, 0.0], 0.016);
        let vel = tracker.update([1.0, 0.0, 0.0], 0.1);
        // With smoothing=1.0, instant velocity: dx/dt = 1.0/0.1 = 10.0
        assert!((vel[0] - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_velocity_tracker_smoothing() {
        let mut tracker = CameraVelocityTracker::new(0.5);
        tracker.update([0.0, 0.0, 0.0], 0.016);
        tracker.update([1.0, 0.0, 0.0], 0.1);
        // Smoothed velocity should be less than instant velocity
        assert!(tracker.speed() < 10.0);
        assert!(tracker.speed() > 0.0);
    }

    #[test]
    fn test_velocity_tracker_reset() {
        let mut tracker = CameraVelocityTracker::new(1.0);
        tracker.update([0.0, 0.0, 0.0], 0.016);
        tracker.update([10.0, 5.0, 0.0], 0.1);
        tracker.reset();
        assert_eq!(tracker.velocity(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_velocity_tracker_zero_delta_time() {
        let mut tracker = CameraVelocityTracker::new(1.0);
        tracker.update([0.0, 0.0, 0.0], 0.016);
        let vel = tracker.update([10.0, 0.0, 0.0], 0.0);
        // Should not change velocity with zero delta time
        assert_eq!(vel, [0.0, 0.0, 0.0]);
    }

    // ---- FrustumPredictor ----

    #[test]
    fn test_predictor_default() {
        let p = FrustumPredictor::default();
        assert_eq!(p.prediction_frames, 10);
    }

    #[test]
    fn test_predictor_prediction_time() {
        let p = FrustumPredictor::new(10, 0.016);
        let time = p.prediction_time();
        assert!((time - 0.16).abs() < 0.001);
    }

    #[test]
    fn test_predictor_position() {
        let p = FrustumPredictor::new(10, 0.1);
        let camera = CameraState::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 60.0);
        let velocity = [10.0, 0.0, 0.0];
        let predicted = p.predict_position(&camera, velocity);
        // After 1 second (10 frames * 0.1s), should move 10 units
        assert!((predicted[0] - 10.0).abs() < 0.001);
    }

    // ---- AssetInfo ----

    #[test]
    fn test_asset_info_creation() {
        let asset = AssetInfo::new(ContentHash::new(1), [0.0, 0.0, 0.0], 1.0);
        assert_eq!(asset.hash, ContentHash::new(1));
        assert_eq!(asset.lod_bias, 0.0);
        assert!(!asset.visible);
    }

    #[test]
    fn test_asset_info_builder() {
        let asset = AssetInfo::new(ContentHash::new(1), [5.0, 0.0, 0.0], 2.0)
            .with_lod_bias(-0.5)
            .with_visible(true)
            .with_load_frames(5.0);
        assert_eq!(asset.lod_bias, -0.5);
        assert!(asset.visible);
        assert_eq!(asset.load_frames, 5.0);
    }

    // ---- PreloadRequest ----

    #[test]
    fn test_request_ordering() {
        let r1 = PreloadRequest::new(ContentHash::new(1), 0.8, 5.0);
        let r2 = PreloadRequest::new(ContentHash::new(2), 0.5, 5.0);
        assert!(r1 > r2); // Higher priority should be greater
    }

    #[test]
    fn test_request_ordering_same_priority() {
        let r1 = PreloadRequest::new(ContentHash::new(1), 0.5, 3.0);
        let r2 = PreloadRequest::new(ContentHash::new(2), 0.5, 5.0);
        assert!(r1 > r2); // Sooner need time should be greater
    }

    // ---- PreloadQueue ----

    #[test]
    fn test_queue_submit() {
        let mut queue = PreloadQueue::new(5, 10);
        let req = PreloadRequest::new(ContentHash::new(1), 0.5, 1.0);
        assert!(queue.submit(req.clone()));
        assert!(!queue.submit(req)); // Duplicate
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn test_queue_cancel() {
        let mut queue = PreloadQueue::new(5, 10);
        queue.submit(PreloadRequest::new(ContentHash::new(1), 0.5, 1.0));
        assert!(queue.cancel(ContentHash::new(1)));
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn test_queue_dequeue() {
        let mut queue = PreloadQueue::new(2, 10);
        queue.submit(PreloadRequest::new(ContentHash::new(1), 0.3, 1.0));
        queue.submit(PreloadRequest::new(ContentHash::new(2), 0.8, 1.0));
        queue.submit(PreloadRequest::new(ContentHash::new(3), 0.5, 1.0));

        let batch = queue.dequeue();
        assert_eq!(batch.len(), 2); // Rate limit = 2
        assert_eq!(batch[0].asset_id, ContentHash::new(2)); // Highest priority first
        assert_eq!(queue.active_count(), 2);
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn test_queue_rate_limiting() {
        let mut queue = PreloadQueue::new(1, 10);
        queue.submit(PreloadRequest::new(ContentHash::new(1), 0.5, 1.0));
        queue.submit(PreloadRequest::new(ContentHash::new(2), 0.8, 1.0));

        let batch = queue.dequeue();
        assert_eq!(batch.len(), 1); // Rate limited to 1
    }

    #[test]
    fn test_queue_max_active() {
        let mut queue = PreloadQueue::new(5, 2);
        queue.submit(PreloadRequest::new(ContentHash::new(1), 0.5, 1.0));
        queue.submit(PreloadRequest::new(ContentHash::new(2), 0.6, 1.0));
        queue.submit(PreloadRequest::new(ContentHash::new(3), 0.7, 1.0));

        let batch1 = queue.dequeue();
        assert_eq!(batch1.len(), 2); // Max active = 2

        let batch2 = queue.dequeue();
        assert_eq!(batch2.len(), 0); // No slots available

        queue.complete(ContentHash::new(3));
        let batch3 = queue.dequeue();
        assert_eq!(batch3.len(), 1); // One slot freed
    }

    #[test]
    fn test_queue_complete() {
        let mut queue = PreloadQueue::new(5, 10);
        queue.submit(PreloadRequest::new(ContentHash::new(1), 0.5, 1.0));
        queue.dequeue();
        assert_eq!(queue.active_count(), 1);
        queue.complete(ContentHash::new(1));
        assert_eq!(queue.active_count(), 0);
    }

    #[test]
    fn test_queue_cancel_all() {
        let mut queue = PreloadQueue::new(5, 10);
        queue.submit(PreloadRequest::new(ContentHash::new(1), 0.5, 1.0));
        queue.submit(PreloadRequest::new(ContentHash::new(2), 0.6, 1.0));
        queue.cancel_all();
        assert_eq!(queue.pending_count(), 0);
    }

    // ---- AssetScorer ----

    #[test]
    fn test_scorer_visible_asset() {
        let scorer = AssetScorer::default();
        let asset = AssetInfo::new(ContentHash::new(1), [10.0, 0.0, 0.0], 1.0)
            .with_visible(true);
        let score = scorer.score(&asset, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(score > 0.0);
    }

    #[test]
    fn test_scorer_moving_towards_asset() {
        let scorer = AssetScorer::default();
        let asset = AssetInfo::new(ContentHash::new(1), [10.0, 0.0, 0.0], 1.0);
        let score_towards = scorer.score(&asset, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let score_away = scorer.score(&asset, [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(score_towards > score_away);
    }

    #[test]
    fn test_scorer_distance_matters() {
        let scorer = AssetScorer::default();
        let near = AssetInfo::new(ContentHash::new(1), [10.0, 0.0, 0.0], 1.0);
        let far = AssetInfo::new(ContentHash::new(2), [100.0, 0.0, 0.0], 1.0);
        let score_near = scorer.score(&near, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let score_far = scorer.score(&far, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(score_near > score_far);
    }

    #[test]
    fn test_scorer_beyond_max_distance() {
        let scorer = AssetScorer::new(PreloadWeights::default(), 100.0);
        let asset = AssetInfo::new(ContentHash::new(1), [200.0, 0.0, 0.0], 1.0);
        let score = scorer.score(&asset, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_scorer_lod_bias() {
        let scorer = AssetScorer::default();
        let high_lod = AssetInfo::new(ContentHash::new(1), [10.0, 0.0, 0.0], 1.0)
            .with_lod_bias(1.0);
        let low_lod = AssetInfo::new(ContentHash::new(2), [10.0, 0.0, 0.0], 1.0)
            .with_lod_bias(-1.0);
        let score_high = scorer.score(&high_lod, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let score_low = scorer.score(&low_lod, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(score_high > score_low);
    }

    // ---- PreloadManager ----

    #[test]
    fn test_manager_creation() {
        let manager = PreloadManager::default();
        assert_eq!(manager.pending_count(), 0);
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_manager_update_camera() {
        let mut manager = PreloadManager::default();
        manager.update_camera(CameraState::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 60.0), 0.016);
        manager.update_camera(CameraState::new([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], 60.0), 0.016);
        let vel = manager.camera_velocity();
        assert!(vel[0] > 0.0);
    }

    #[test]
    fn test_manager_predict_position() {
        let mut manager = PreloadManager::with_config(PreloadWeights::default(), 10, 5, 10);
        manager.update_camera(CameraState::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 60.0), 0.016);
        manager.update_camera(CameraState::new([10.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0), 1.0);
        let predicted = manager.predicted_position();
        assert!(predicted[0] > 10.0); // Should extrapolate forward
    }

    #[test]
    fn test_manager_predict_and_queue() {
        let mut manager = PreloadManager::default();
        let assets = vec![
            AssetInfo::new(ContentHash::new(1), [10.0, 0.0, 0.0], 1.0).with_visible(true),
            AssetInfo::new(ContentHash::new(2), [20.0, 0.0, 0.0], 1.0),
        ];

        let submitted = manager.predict_and_queue(&assets, 5);
        assert!(submitted > 0);
        assert!(manager.pending_count() > 0);
    }

    #[test]
    fn test_manager_predict_empty_assets() {
        let mut manager = PreloadManager::default();
        let submitted = manager.predict_and_queue(&[], 5);
        assert_eq!(submitted, 0);
    }

    #[test]
    fn test_manager_dequeue_and_complete() {
        let mut manager = PreloadManager::default();
        let assets = vec![
            AssetInfo::new(ContentHash::new(1), [10.0, 0.0, 0.0], 1.0).with_visible(true),
        ];

        manager.predict_and_queue(&assets, 5);
        let requests = manager.dequeue();
        assert_eq!(requests.len(), 1);
        assert_eq!(manager.active_count(), 1);

        manager.complete(requests[0].asset_id);
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_manager_cancel() {
        let mut manager = PreloadManager::default();
        let assets = vec![
            AssetInfo::new(ContentHash::new(1), [10.0, 0.0, 0.0], 1.0).with_visible(true),
        ];

        manager.predict_and_queue(&assets, 5);
        assert!(manager.cancel(ContentHash::new(1)));
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_manager_reset() {
        let mut manager = PreloadManager::default();
        manager.update_camera(CameraState::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 60.0), 0.016);
        manager.update_camera(CameraState::new([10.0, 0.0, 0.0], [0.0, 0.0, -1.0], 60.0), 0.016);

        let assets = vec![
            AssetInfo::new(ContentHash::new(1), [10.0, 0.0, 0.0], 1.0).with_visible(true),
        ];
        manager.predict_and_queue(&assets, 5);

        manager.reset();
        assert_eq!(manager.camera_velocity(), [0.0, 0.0, 0.0]);
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_manager_priority_ordering() {
        let mut manager = PreloadManager::default();
        let assets = vec![
            AssetInfo::new(ContentHash::new(1), [100.0, 0.0, 0.0], 1.0),
            AssetInfo::new(ContentHash::new(2), [10.0, 0.0, 0.0], 1.0).with_visible(true),
            AssetInfo::new(ContentHash::new(3), [50.0, 0.0, 0.0], 1.0),
        ];

        manager.predict_and_queue(&assets, 3);
        let requests = manager.dequeue();

        // Asset 2 should be first (visible + closer)
        assert_eq!(requests[0].asset_id, ContentHash::new(2));
    }
}
