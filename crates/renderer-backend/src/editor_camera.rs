//! Editor camera with orbit, pan, zoom, and fly controls (T-TL-2.1).
//!
//! Provides a full-featured camera for 3D viewport navigation in the editor.
//! Supports multiple interaction modes:
//! - **Orbit**: Rotate around a target point (Alt+LMB drag)
//! - **Pan**: Translate target and camera parallel to view plane (MMB drag or Alt+MMB)
//! - **Zoom**: Dolly towards/away from target (scroll wheel)
//! - **Fly**: WASD movement + mouse look (RMB held)

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

/// Camera interaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraMode {
    /// Orbit around the target point.
    #[default]
    Orbit,
    /// Free-flying camera with WASD + mouse look.
    Fly,
    /// Pan the view parallel to the view plane.
    Pan,
}

/// Saved camera pose for save/restore functionality.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CameraPose {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
}

impl Default for CameraPose {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 5.0, 10.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y: 45.0_f32.to_radians(),
        }
    }
}

/// Full-featured editor camera with orbit, pan, zoom, and fly controls.
#[derive(Debug, Clone)]
pub struct EditorCamera {
    /// Camera position in world space.
    pub position: Vec3,
    /// Look-at target point (used for orbit mode).
    pub target: Vec3,
    /// Up direction (usually +Y).
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// Current camera interaction mode.
    mode: CameraMode,
    /// Orbit sensitivity multiplier.
    orbit_sensitivity: f32,
    /// Pan sensitivity multiplier.
    pan_sensitivity: f32,
    /// Zoom sensitivity multiplier.
    zoom_sensitivity: f32,
    /// Fly movement speed.
    fly_speed: f32,
    /// Fly look sensitivity.
    fly_look_sensitivity: f32,
    /// Minimum zoom distance (prevents getting too close to target).
    min_distance: f32,
    /// Maximum zoom distance.
    max_distance: f32,
    /// Minimum pitch angle in radians (prevents gimbal lock).
    min_pitch: f32,
    /// Maximum pitch angle in radians.
    max_pitch: f32,
}

impl Default for EditorCamera {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorCamera {
    /// Create a new editor camera with default settings.
    ///
    /// Default position is (0, 5, 10) looking at origin.
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 5.0, 10.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y: 45.0_f32.to_radians(),
            near: 0.1,
            far: 1000.0,
            aspect: 16.0 / 9.0,
            mode: CameraMode::Orbit,
            orbit_sensitivity: 0.01,
            pan_sensitivity: 0.01,
            zoom_sensitivity: 0.1,
            fly_speed: 10.0,
            fly_look_sensitivity: 0.003,
            min_distance: 0.1,
            max_distance: 10000.0,
            min_pitch: -89.0_f32.to_radians(),
            max_pitch: 89.0_f32.to_radians(),
        }
    }

    /// Create a camera with custom initial position and target.
    pub fn with_position_target(position: Vec3, target: Vec3) -> Self {
        Self {
            position,
            target,
            ..Self::new()
        }
    }

    /// Get the current camera mode.
    pub fn mode(&self) -> CameraMode {
        self.mode
    }

    /// Set the camera mode.
    pub fn set_mode(&mut self, mode: CameraMode) {
        self.mode = mode;
    }

    /// Get the distance from camera to target.
    pub fn distance(&self) -> f32 {
        (self.position - self.target).length()
    }

    /// Get the forward direction (normalized).
    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize_or_zero()
    }

    /// Get the right direction (normalized).
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize_or_zero()
    }

    /// Get the camera's local up direction (perpendicular to forward and right).
    pub fn local_up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    /// Orbit the camera around the target point.
    ///
    /// `delta_x` rotates horizontally (yaw), `delta_y` rotates vertically (pitch).
    /// Positive `delta_x` rotates right, positive `delta_y` rotates up.
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        let yaw = -delta_x * self.orbit_sensitivity;
        let pitch = -delta_y * self.orbit_sensitivity;

        // Vector from target to camera
        let mut offset = self.position - self.target;
        let distance = offset.length();

        if distance < f32::EPSILON {
            return;
        }

        // Convert to spherical coordinates
        let mut theta = offset.x.atan2(offset.z); // Azimuth (yaw)
        let mut phi = (offset.y / distance).asin(); // Elevation (pitch)

        // Apply rotation deltas
        theta += yaw;
        phi += pitch;

        // Clamp pitch to avoid gimbal lock
        phi = phi.clamp(self.min_pitch, self.max_pitch);

        // Convert back to Cartesian
        let cos_phi = phi.cos();
        offset.x = distance * cos_phi * theta.sin();
        offset.y = distance * phi.sin();
        offset.z = distance * cos_phi * theta.cos();

        self.position = self.target + offset;
    }

    /// Pan the camera parallel to the view plane.
    ///
    /// `delta_x` moves right, `delta_y` moves up.
    /// Both camera position and target are translated together.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let distance = self.distance();
        let scale = distance * self.pan_sensitivity;

        let right = self.right();
        let up = self.local_up();

        let offset = right * (-delta_x * scale) + up * (delta_y * scale);

        self.position += offset;
        self.target += offset;
    }

    /// Zoom (dolly) towards or away from the target.
    ///
    /// Positive `delta` zooms in (moves towards target),
    /// negative `delta` zooms out (moves away from target).
    pub fn zoom(&mut self, delta: f32) {
        let current_distance = self.distance();

        // Calculate zoom amount as a fraction of current distance
        let zoom_amount = current_distance * delta * self.zoom_sensitivity;
        let new_distance = (current_distance - zoom_amount).clamp(self.min_distance, self.max_distance);

        // Move camera along the view direction
        let direction = (self.position - self.target).normalize_or_zero();
        self.position = self.target + direction * new_distance;
    }

    /// Move the camera in fly mode.
    ///
    /// `forward` moves along the view direction (positive = forward).
    /// `right` moves along the right direction (positive = right).
    /// `up` moves along the world up direction (positive = up).
    ///
    /// The values are typically in range [-1, 1] and scaled by fly_speed.
    pub fn fly_move(&mut self, forward: f32, right: f32, up: f32) {
        let forward_dir = self.forward();
        let right_dir = self.right();

        let movement = forward_dir * (forward * self.fly_speed)
            + right_dir * (right * self.fly_speed)
            + Vec3::Y * (up * self.fly_speed);

        self.position += movement;
        self.target += movement;
    }

    /// Look around in fly mode.
    ///
    /// `yaw` rotates horizontally (positive = look right).
    /// `pitch` rotates vertically (positive = look up).
    pub fn fly_look(&mut self, yaw: f32, pitch: f32) {
        let yaw_delta = -yaw * self.fly_look_sensitivity;
        let pitch_delta = -pitch * self.fly_look_sensitivity;

        // Get current direction
        let mut direction = self.target - self.position;
        let distance = direction.length();

        if distance < f32::EPSILON {
            return;
        }

        direction = direction.normalize();

        // Calculate current angles
        let mut theta = direction.x.atan2(direction.z); // Azimuth
        let mut phi = direction.y.asin(); // Elevation

        // Apply deltas
        theta += yaw_delta;
        phi += pitch_delta;

        // Clamp pitch
        phi = phi.clamp(self.min_pitch, self.max_pitch);

        // Convert back to direction
        let cos_phi = phi.cos();
        direction.x = cos_phi * theta.sin();
        direction.y = phi.sin();
        direction.z = cos_phi * theta.cos();

        self.target = self.position + direction * distance;
    }

    /// Focus the camera on a target point at a specified distance.
    ///
    /// The camera smoothly repositions to look at the target from
    /// the current viewing direction.
    pub fn focus_on(&mut self, target: Vec3, distance: f32) {
        let clamped_distance = distance.clamp(self.min_distance, self.max_distance);

        // Keep current viewing direction if possible
        let direction = if self.distance() > f32::EPSILON {
            (self.position - self.target).normalize()
        } else {
            Vec3::new(0.0, 0.5, 1.0).normalize()
        };

        self.target = target;
        self.position = target + direction * clamped_distance;
    }

    /// Compute the view matrix (world-to-camera transform).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Compute the projection matrix (perspective).
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Compute the combined view-projection matrix.
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Save the current camera pose.
    pub fn save_pose(&self) -> CameraPose {
        CameraPose {
            position: self.position,
            target: self.target,
            up: self.up,
            fov_y: self.fov_y,
        }
    }

    /// Restore the camera from a saved pose.
    pub fn restore_pose(&mut self, pose: &CameraPose) {
        self.position = pose.position;
        self.target = pose.target;
        self.up = pose.up;
        self.fov_y = pose.fov_y;
    }

    /// Set the orbit sensitivity multiplier.
    pub fn set_orbit_sensitivity(&mut self, sensitivity: f32) {
        self.orbit_sensitivity = sensitivity.max(0.0001);
    }

    /// Set the pan sensitivity multiplier.
    pub fn set_pan_sensitivity(&mut self, sensitivity: f32) {
        self.pan_sensitivity = sensitivity.max(0.0001);
    }

    /// Set the zoom sensitivity multiplier.
    pub fn set_zoom_sensitivity(&mut self, sensitivity: f32) {
        self.zoom_sensitivity = sensitivity.max(0.0001);
    }

    /// Set the fly movement speed.
    pub fn set_fly_speed(&mut self, speed: f32) {
        self.fly_speed = speed.max(0.0);
    }

    /// Set the fly look sensitivity.
    pub fn set_fly_look_sensitivity(&mut self, sensitivity: f32) {
        self.fly_look_sensitivity = sensitivity.max(0.0001);
    }

    /// Set the minimum zoom distance.
    pub fn set_min_distance(&mut self, distance: f32) {
        self.min_distance = distance.max(0.001);
    }

    /// Set the maximum zoom distance.
    pub fn set_max_distance(&mut self, distance: f32) {
        self.max_distance = distance.max(self.min_distance + 1.0);
    }

    /// Get the frustum planes for culling (left, right, bottom, top, near, far).
    /// Each plane is represented as (normal, distance) where distance is from origin.
    pub fn frustum_planes(&self) -> [Vec3; 6] {
        let vp = self.view_projection_matrix();

        // Extract frustum planes from view-projection matrix
        // Plane equation: dot(normal, point) + d = 0
        let row0 = Vec3::new(vp.col(0).x, vp.col(1).x, vp.col(2).x);
        let row1 = Vec3::new(vp.col(0).y, vp.col(1).y, vp.col(2).y);
        let row2 = Vec3::new(vp.col(0).z, vp.col(1).z, vp.col(2).z);
        let row3 = Vec3::new(vp.col(0).w, vp.col(1).w, vp.col(2).w);

        [
            (row3 + row0).normalize_or_zero(), // Left
            (row3 - row0).normalize_or_zero(), // Right
            (row3 + row1).normalize_or_zero(), // Bottom
            (row3 - row1).normalize_or_zero(), // Top
            (row3 + row2).normalize_or_zero(), // Near
            (row3 - row2).normalize_or_zero(), // Far
        ]
    }

    /// Reset the camera to default state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Set the aspect ratio from viewport dimensions.
    pub fn set_viewport_size(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    // === Construction Tests ===

    #[test]
    fn test_new_creates_default_camera() {
        let cam = EditorCamera::new();
        assert!(vec3_approx_eq(cam.position, Vec3::new(0.0, 5.0, 10.0)));
        assert!(vec3_approx_eq(cam.target, Vec3::ZERO));
        assert!(vec3_approx_eq(cam.up, Vec3::Y));
        assert_eq!(cam.mode(), CameraMode::Orbit);
    }

    #[test]
    fn test_default_matches_new() {
        let cam1 = EditorCamera::new();
        let cam2 = EditorCamera::default();
        assert!(vec3_approx_eq(cam1.position, cam2.position));
        assert!(vec3_approx_eq(cam1.target, cam2.target));
    }

    #[test]
    fn test_with_position_target() {
        let pos = Vec3::new(5.0, 5.0, 5.0);
        let target = Vec3::new(0.0, 1.0, 0.0);
        let cam = EditorCamera::with_position_target(pos, target);
        assert!(vec3_approx_eq(cam.position, pos));
        assert!(vec3_approx_eq(cam.target, target));
    }

    // === Mode Tests ===

    #[test]
    fn test_set_mode() {
        let mut cam = EditorCamera::new();
        assert_eq!(cam.mode(), CameraMode::Orbit);

        cam.set_mode(CameraMode::Fly);
        assert_eq!(cam.mode(), CameraMode::Fly);

        cam.set_mode(CameraMode::Pan);
        assert_eq!(cam.mode(), CameraMode::Pan);
    }

    // === Direction Tests ===

    #[test]
    fn test_forward_direction() {
        let cam = EditorCamera::new();
        let forward = cam.forward();
        // Camera at (0, 5, 10) looking at (0, 0, 0)
        // Forward should be roughly (0, -0.447, -0.894)
        assert!(forward.z < 0.0, "Forward should be negative Z");
        assert!(approx_eq(forward.length(), 1.0));
    }

    #[test]
    fn test_right_direction() {
        let cam = EditorCamera::new();
        let right = cam.right();
        // Right should be along positive X axis
        assert!(right.x > 0.9);
        assert!(approx_eq(right.length(), 1.0));
    }

    #[test]
    fn test_local_up_direction() {
        let cam = EditorCamera::new();
        let up = cam.local_up();
        // Local up should be roughly along Y but tilted
        assert!(approx_eq(up.length(), 1.0));
    }

    #[test]
    fn test_distance() {
        let cam = EditorCamera::new();
        // Distance from (0, 5, 10) to (0, 0, 0)
        let expected = (5.0_f32.powi(2) + 10.0_f32.powi(2)).sqrt();
        assert!(approx_eq(cam.distance(), expected));
    }

    // === Orbit Tests ===

    #[test]
    fn test_orbit_horizontal() {
        let mut cam = EditorCamera::new();
        let initial_pos = cam.position;
        let initial_distance = cam.distance();

        cam.orbit(100.0, 0.0); // Rotate horizontally

        // Distance should be preserved
        assert!(approx_eq(cam.distance(), initial_distance));
        // Position should have changed
        assert!(!vec3_approx_eq(cam.position, initial_pos));
        // Target should be unchanged
        assert!(vec3_approx_eq(cam.target, Vec3::ZERO));
    }

    #[test]
    fn test_orbit_vertical() {
        let mut cam = EditorCamera::new();
        let initial_distance = cam.distance();

        cam.orbit(0.0, 50.0); // Rotate vertically

        // Distance should be preserved
        assert!(approx_eq(cam.distance(), initial_distance));
    }

    #[test]
    fn test_orbit_combined() {
        let mut cam = EditorCamera::new();
        let initial_distance = cam.distance();

        cam.orbit(50.0, 30.0);

        assert!(approx_eq(cam.distance(), initial_distance));
    }

    #[test]
    fn test_orbit_preserves_target() {
        let mut cam = EditorCamera::new();
        let target = cam.target;

        cam.orbit(100.0, 50.0);

        assert!(vec3_approx_eq(cam.target, target));
    }

    #[test]
    fn test_orbit_pitch_clamped() {
        let mut cam = EditorCamera::new();

        // Try to orbit past vertical limit
        for _ in 0..100 {
            cam.orbit(0.0, 100.0);
        }

        // Camera should not be directly above target (gimbal lock prevention)
        let direction = (cam.position - cam.target).normalize();
        assert!(direction.y < 0.999, "Pitch should be clamped");
    }

    // === Pan Tests ===

    #[test]
    fn test_pan_horizontal() {
        let mut cam = EditorCamera::new();
        let initial_pos = cam.position;
        let initial_target = cam.target;

        cam.pan(100.0, 0.0);

        // Both position and target should move
        let pos_delta = cam.position - initial_pos;
        let target_delta = cam.target - initial_target;

        // They should move the same amount (parallel translation)
        assert!(vec3_approx_eq(pos_delta, target_delta));
    }

    #[test]
    fn test_pan_vertical() {
        let mut cam = EditorCamera::new();
        let initial_distance = cam.distance();

        cam.pan(0.0, 100.0);

        // Distance should be preserved
        assert!(approx_eq(cam.distance(), initial_distance));
    }

    #[test]
    fn test_pan_preserves_distance() {
        let mut cam = EditorCamera::new();
        let initial_distance = cam.distance();

        cam.pan(50.0, 50.0);

        assert!(approx_eq(cam.distance(), initial_distance));
    }

    #[test]
    fn test_pan_preserves_view_direction() {
        let mut cam = EditorCamera::new();
        let initial_forward = cam.forward();

        cam.pan(50.0, 50.0);

        let new_forward = cam.forward();
        assert!(vec3_approx_eq(initial_forward, new_forward));
    }

    // === Zoom Tests ===

    #[test]
    fn test_zoom_in() {
        let mut cam = EditorCamera::new();
        let initial_distance = cam.distance();

        cam.zoom(1.0); // Zoom in

        assert!(cam.distance() < initial_distance);
    }

    #[test]
    fn test_zoom_out() {
        let mut cam = EditorCamera::new();
        let initial_distance = cam.distance();

        cam.zoom(-1.0); // Zoom out

        assert!(cam.distance() > initial_distance);
    }

    #[test]
    fn test_zoom_preserves_target() {
        let mut cam = EditorCamera::new();
        let target = cam.target;

        cam.zoom(1.0);

        assert!(vec3_approx_eq(cam.target, target));
    }

    #[test]
    fn test_zoom_min_distance_limit() {
        let mut cam = EditorCamera::new();
        cam.set_min_distance(5.0);

        // Zoom in a lot
        for _ in 0..100 {
            cam.zoom(10.0);
        }

        assert!(cam.distance() >= 5.0);
    }

    #[test]
    fn test_zoom_max_distance_limit() {
        let mut cam = EditorCamera::new();
        cam.set_max_distance(100.0);

        // Zoom out a lot
        for _ in 0..100 {
            cam.zoom(-10.0);
        }

        assert!(cam.distance() <= 100.0);
    }

    // === Fly Mode Tests ===

    #[test]
    fn test_fly_move_forward() {
        let mut cam = EditorCamera::new();
        let initial_pos = cam.position;

        cam.fly_move(1.0, 0.0, 0.0);

        // Should move along forward direction
        let movement = cam.position - initial_pos;
        assert!(movement.length() > 0.0);
    }

    #[test]
    fn test_fly_move_right() {
        let mut cam = EditorCamera::new();
        let initial_pos = cam.position;

        cam.fly_move(0.0, 1.0, 0.0);

        // Should move along right direction
        assert!(cam.position.x > initial_pos.x);
    }

    #[test]
    fn test_fly_move_up() {
        let mut cam = EditorCamera::new();
        let initial_pos = cam.position;

        cam.fly_move(0.0, 0.0, 1.0);

        // Should move up
        assert!(cam.position.y > initial_pos.y);
    }

    #[test]
    fn test_fly_move_preserves_distance() {
        let mut cam = EditorCamera::new();
        let initial_distance = cam.distance();

        cam.fly_move(1.0, 0.5, 0.2);

        // Distance should remain the same (target moves with camera)
        assert!(approx_eq(cam.distance(), initial_distance));
    }

    #[test]
    fn test_fly_look_yaw() {
        let mut cam = EditorCamera::new();
        let initial_target = cam.target;

        cam.fly_look(100.0, 0.0);

        // Target should change but position should stay
        assert!(!vec3_approx_eq(cam.target, initial_target));
    }

    #[test]
    fn test_fly_look_pitch() {
        let mut cam = EditorCamera::new();
        let initial_target = cam.target;

        cam.fly_look(0.0, 100.0);

        assert!(!vec3_approx_eq(cam.target, initial_target));
    }

    #[test]
    fn test_fly_look_preserves_distance() {
        let mut cam = EditorCamera::new();
        let initial_distance = cam.distance();

        cam.fly_look(100.0, 50.0);

        assert!(approx_eq(cam.distance(), initial_distance));
    }

    // === Focus On Tests ===

    #[test]
    fn test_focus_on_changes_target() {
        let mut cam = EditorCamera::new();
        let new_target = Vec3::new(10.0, 5.0, 3.0);

        cam.focus_on(new_target, 5.0);

        assert!(vec3_approx_eq(cam.target, new_target));
    }

    #[test]
    fn test_focus_on_sets_distance() {
        let mut cam = EditorCamera::new();
        let new_target = Vec3::new(10.0, 5.0, 3.0);

        cam.focus_on(new_target, 5.0);

        assert!(approx_eq(cam.distance(), 5.0));
    }

    #[test]
    fn test_focus_on_respects_min_distance() {
        let mut cam = EditorCamera::new();
        cam.set_min_distance(2.0);

        cam.focus_on(Vec3::ZERO, 0.5);

        assert!(cam.distance() >= 2.0);
    }

    #[test]
    fn test_focus_on_respects_max_distance() {
        let mut cam = EditorCamera::new();
        cam.set_max_distance(100.0);

        cam.focus_on(Vec3::ZERO, 500.0);

        // Allow small floating point error
        assert!(cam.distance() <= 100.0 + EPSILON);
    }

    // === Matrix Tests ===

    #[test]
    fn test_view_matrix_not_identity() {
        let cam = EditorCamera::new();
        let view = cam.view_matrix();

        assert_ne!(view, Mat4::IDENTITY);
    }

    #[test]
    fn test_projection_matrix_not_identity() {
        let cam = EditorCamera::new();
        let proj = cam.projection_matrix();

        assert_ne!(proj, Mat4::IDENTITY);
    }

    #[test]
    fn test_view_projection_matrix() {
        let cam = EditorCamera::new();
        let vp = cam.view_projection_matrix();
        let expected = cam.projection_matrix() * cam.view_matrix();

        // Matrices should be equal
        for i in 0..4 {
            for j in 0..4 {
                assert!(approx_eq(vp.col(i)[j], expected.col(i)[j]));
            }
        }
    }

    #[test]
    fn test_view_matrix_transforms_target_to_center() {
        let cam = EditorCamera::new();
        let view = cam.view_matrix();

        // Target should transform to somewhere on negative Z axis
        let target_view = view.transform_point3(cam.target);
        assert!(approx_eq(target_view.x, 0.0));
        assert!(approx_eq(target_view.y, 0.0));
        assert!(target_view.z < 0.0);
    }

    // === Pose Save/Restore Tests ===

    #[test]
    fn test_save_pose() {
        let cam = EditorCamera::new();
        let pose = cam.save_pose();

        assert!(vec3_approx_eq(pose.position, cam.position));
        assert!(vec3_approx_eq(pose.target, cam.target));
        assert!(vec3_approx_eq(pose.up, cam.up));
        assert!(approx_eq(pose.fov_y, cam.fov_y));
    }

    #[test]
    fn test_restore_pose() {
        let mut cam = EditorCamera::new();
        let pose = CameraPose {
            position: Vec3::new(1.0, 2.0, 3.0),
            target: Vec3::new(4.0, 5.0, 6.0),
            up: Vec3::Y,
            fov_y: 60.0_f32.to_radians(),
        };

        cam.restore_pose(&pose);

        assert!(vec3_approx_eq(cam.position, pose.position));
        assert!(vec3_approx_eq(cam.target, pose.target));
        assert!(vec3_approx_eq(cam.up, pose.up));
        assert!(approx_eq(cam.fov_y, pose.fov_y));
    }

    #[test]
    fn test_pose_roundtrip() {
        let mut cam = EditorCamera::new();

        // Modify camera
        cam.orbit(50.0, 30.0);
        cam.zoom(1.5);
        cam.pan(10.0, 5.0);

        // Save and restore
        let pose = cam.save_pose();
        let mut cam2 = EditorCamera::new();
        cam2.restore_pose(&pose);

        assert!(vec3_approx_eq(cam.position, cam2.position));
        assert!(vec3_approx_eq(cam.target, cam2.target));
    }

    #[test]
    fn test_pose_serialization() {
        let pose = CameraPose::default();
        let json = serde_json::to_string(&pose).unwrap();
        let restored: CameraPose = serde_json::from_str(&json).unwrap();

        assert_eq!(pose, restored);
    }

    // === Sensitivity Tests ===

    #[test]
    fn test_set_orbit_sensitivity() {
        let mut cam = EditorCamera::new();
        cam.set_orbit_sensitivity(0.02);

        // Orbit with same delta should now move more
        let initial_pos = cam.position;
        cam.orbit(100.0, 0.0);
        let movement = (cam.position - initial_pos).length();

        let mut cam2 = EditorCamera::new();
        cam2.set_orbit_sensitivity(0.01);
        let initial_pos2 = cam2.position;
        cam2.orbit(100.0, 0.0);
        let movement2 = (cam2.position - initial_pos2).length();

        assert!(movement > movement2);
    }

    #[test]
    fn test_set_fly_speed() {
        let mut cam = EditorCamera::new();
        cam.set_fly_speed(20.0);

        let initial_pos = cam.position;
        cam.fly_move(1.0, 0.0, 0.0);
        let movement = (cam.position - initial_pos).length();

        assert!(movement > 15.0);
    }

    // === Reset Tests ===

    #[test]
    fn test_reset() {
        let mut cam = EditorCamera::new();
        cam.orbit(100.0, 50.0);
        cam.zoom(2.0);
        cam.pan(50.0, 50.0);
        cam.set_mode(CameraMode::Fly);

        cam.reset();

        let default_cam = EditorCamera::new();
        assert!(vec3_approx_eq(cam.position, default_cam.position));
        assert!(vec3_approx_eq(cam.target, default_cam.target));
        assert_eq!(cam.mode(), default_cam.mode());
    }

    // === Viewport Tests ===

    #[test]
    fn test_set_viewport_size() {
        let mut cam = EditorCamera::new();
        cam.set_viewport_size(1920, 1080);

        assert!(approx_eq(cam.aspect, 1920.0 / 1080.0));
    }

    #[test]
    fn test_set_viewport_size_zero_height() {
        let mut cam = EditorCamera::new();
        let original_aspect = cam.aspect;
        cam.set_viewport_size(1920, 0);

        // Should not change on zero height
        assert!(approx_eq(cam.aspect, original_aspect));
    }

    // === Frustum Tests ===

    #[test]
    fn test_frustum_planes_count() {
        let cam = EditorCamera::new();
        let planes = cam.frustum_planes();

        assert_eq!(planes.len(), 6);
    }

    #[test]
    fn test_frustum_planes_normalized() {
        let cam = EditorCamera::new();
        let planes = cam.frustum_planes();

        for plane in &planes {
            // All planes should be normalized (or zero)
            let len = plane.length();
            assert!(len < 0.001 || approx_eq(len, 1.0));
        }
    }

    // === Edge Case Tests ===

    #[test]
    fn test_camera_at_target() {
        let mut cam = EditorCamera::new();
        cam.position = cam.target;

        // Operations should not panic
        cam.orbit(10.0, 10.0);
        cam.zoom(1.0);
        cam.fly_look(10.0, 10.0);
    }

    #[test]
    fn test_very_small_movements() {
        let mut cam = EditorCamera::new();
        let initial_pos = cam.position;

        cam.orbit(0.0001, 0.0001);
        cam.pan(0.0001, 0.0001);
        cam.zoom(0.0001);

        // Should not cause NaN or instability
        assert!(!cam.position.is_nan());
        assert!(!cam.target.is_nan());
    }

    #[test]
    fn test_large_movements() {
        let mut cam = EditorCamera::new();

        cam.orbit(10000.0, 10000.0);
        cam.pan(10000.0, 10000.0);
        cam.zoom(100.0);

        // Should not cause NaN or instability
        assert!(!cam.position.is_nan());
        assert!(!cam.target.is_nan());
        assert!(cam.distance().is_finite());
    }

    #[test]
    fn test_negative_sensitivity_clamped() {
        let mut cam = EditorCamera::new();
        cam.set_orbit_sensitivity(-1.0);
        cam.set_pan_sensitivity(-1.0);
        cam.set_zoom_sensitivity(-1.0);
        cam.set_fly_speed(-1.0);

        // Should clamp to positive values
        cam.orbit(100.0, 100.0);
        cam.pan(100.0, 100.0);
        cam.zoom(1.0);

        assert!(!cam.position.is_nan());
    }
}
