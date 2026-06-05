//! Moldable Views for Inspector Panel
//!
//! This module provides type-specific visualization templates for the inspector.
//! Each moldable view knows how to render a specific component type with
//! appropriate UI widgets (color pickers, sliders, gizmo toggles, etc.).
//!
//! # Architecture
//!
//! ```text
//! MoldableRegistry
//! ================
//!     │
//!     ├── TransformView  ─► Position/Rotation/Scale with gizmo toggles
//!     ├── MaterialView   ─► PBR properties, color pickers, textures
//!     ├── MeshView       ─► Vertex/triangle counts, bounds, LOD
//!     ├── CameraView     ─► FOV, near/far, projection, frustum
//!     ├── LightView      ─► Color, intensity, range, shadows
//!     └── PhysicsView    ─► Mass, velocity, collision shape
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use renderer_backend::moldable_views::{MoldableRegistry, TransformView};
//! use renderer_backend::egui_adapter::MockUIContext;
//!
//! let mut registry = MoldableRegistry::new();
//! registry.register_transform();
//!
//! let mut ctx = MockUIContext::new(1);
//! let transform_data = vec![0u8; 40]; // Position + Rotation + Scale
//!
//! if let Some(modified) = registry.render(&mut ctx, "Transform", &transform_data) {
//!     // Apply modified data back to component
//! }
//! ```

use crate::egui_adapter::UIContext;
use crate::inspector_panel::TypeDecoder;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// MoldableView Trait
// ---------------------------------------------------------------------------

/// A specialized view for rendering a specific component type.
///
/// Moldable views provide type-specific UI for component inspection and editing.
/// They know how to decode the raw bytes of a component and render appropriate
/// widgets (sliders, color pickers, dropdowns, etc.).
///
/// Note: This trait uses a concrete type parameter `T: UIContext` instead of
/// `dyn UIContext` because UIContext has generic methods that make it not
/// dyn-compatible (object-safe).
pub trait MoldableView: Send + Sync {
    /// Check if this view can render the given component.
    ///
    /// # Arguments
    ///
    /// * `component_name` - The name of the component type (e.g., "Transform").
    /// * `data` - The raw bytes of the component data.
    ///
    /// # Returns
    ///
    /// `true` if this view can handle the component, `false` otherwise.
    fn can_render(&self, component_name: &str, data: &[u8]) -> bool;

    /// Render the component with this view using MockUIContext.
    ///
    /// This method is provided for testing and concrete implementations.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The UI context to render to.
    /// * `component_name` - The name of the component type.
    /// * `data` - The raw bytes of the component data.
    ///
    /// # Returns
    ///
    /// `Some(modified_bytes)` if the data was modified, `None` otherwise.
    fn render_mock(
        &mut self,
        ctx: &mut crate::egui_adapter::MockUIContext,
        component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>>;

    /// Get the name of this view.
    fn name(&self) -> &str;

    /// Get the priority of this view (higher = more specific).
    ///
    /// When multiple views can render a component, the one with highest
    /// priority is chosen.
    fn priority(&self) -> i32 {
        0
    }
}

// ---------------------------------------------------------------------------
// Coordinate Space
// ---------------------------------------------------------------------------

/// Coordinate space for transform editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CoordinateSpace {
    /// Local space (relative to parent).
    #[default]
    Local,
    /// World space (absolute).
    World,
}

// ---------------------------------------------------------------------------
// Gizmo Mode
// ---------------------------------------------------------------------------

/// Gizmo mode for transform manipulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GizmoMode {
    /// Translate gizmo.
    #[default]
    Translate,
    /// Rotate gizmo.
    Rotate,
    /// Scale gizmo.
    Scale,
}

// ---------------------------------------------------------------------------
// Transform View
// ---------------------------------------------------------------------------

/// Specialized view for Transform components.
///
/// Renders position, rotation, and scale with:
/// - Vec3 sliders for position and scale
/// - Euler angle or quaternion display for rotation
/// - Local/world space toggle
/// - Gizmo mode selector (translate/rotate/scale)
pub struct TransformView {
    /// Current coordinate space.
    space: CoordinateSpace,
    /// Current gizmo mode.
    gizmo_mode: GizmoMode,
    /// Show rotation as euler angles instead of quaternion.
    show_euler: bool,
    /// Snap settings enabled.
    snap_enabled: bool,
    /// Position snap increment.
    snap_position: f32,
    /// Rotation snap increment (degrees).
    snap_rotation: f32,
    /// Scale snap increment.
    snap_scale: f32,
}

impl TransformView {
    /// Create a new transform view with default settings.
    pub fn new() -> Self {
        Self {
            space: CoordinateSpace::Local,
            gizmo_mode: GizmoMode::Translate,
            show_euler: true,
            snap_enabled: false,
            snap_position: 1.0,
            snap_rotation: 15.0,
            snap_scale: 0.1,
        }
    }

    /// Set the coordinate space.
    pub fn set_space(&mut self, space: CoordinateSpace) {
        self.space = space;
    }

    /// Set the gizmo mode.
    pub fn set_gizmo_mode(&mut self, mode: GizmoMode) {
        self.gizmo_mode = mode;
    }

    /// Toggle euler angle display.
    pub fn set_show_euler(&mut self, show: bool) {
        self.show_euler = show;
    }

    /// Get the current coordinate space.
    pub fn space(&self) -> CoordinateSpace {
        self.space
    }

    /// Get the current gizmo mode.
    pub fn gizmo_mode(&self) -> GizmoMode {
        self.gizmo_mode
    }

    /// Check if showing euler angles.
    pub fn show_euler(&self) -> bool {
        self.show_euler
    }

    /// Convert quaternion to euler angles (in degrees).
    fn quat_to_euler(&self, q: [f32; 4]) -> [f32; 3] {
        let (x, y, z, w) = (q[0], q[1], q[2], q[3]);

        // Roll (x-axis rotation)
        let sinr_cosp = 2.0 * (w * x + y * z);
        let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (y-axis rotation)
        let sinp = 2.0 * (w * y - z * x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        // Yaw (z-axis rotation)
        let siny_cosp = 2.0 * (w * z + x * y);
        let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()]
    }

    /// Convert euler angles (in degrees) to quaternion.
    fn euler_to_quat(&self, euler: [f32; 3]) -> [f32; 4] {
        let roll = euler[0].to_radians();
        let pitch = euler[1].to_radians();
        let yaw = euler[2].to_radians();

        let cr = (roll * 0.5).cos();
        let sr = (roll * 0.5).sin();
        let cp = (pitch * 0.5).cos();
        let sp = (pitch * 0.5).sin();
        let cy = (yaw * 0.5).cos();
        let sy = (yaw * 0.5).sin();

        [
            sr * cp * cy - cr * sp * sy, // x
            cr * sp * cy + sr * cp * sy, // y
            cr * cp * sy - sr * sp * cy, // z
            cr * cp * cy + sr * sp * sy, // w
        ]
    }

    /// Apply snap to value if enabled.
    fn apply_snap(&self, value: f32, snap: f32) -> f32 {
        if self.snap_enabled {
            (value / snap).round() * snap
        } else {
            value
        }
    }

    /// Render transform component with any UIContext.
    pub fn render<T: UIContext>(
        &mut self,
        ctx: &mut T,
        _component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        if data.len() < 28 {
            ctx.label("Invalid Transform data");
            return None;
        }

        let mut modified = false;
        let mut result = data.to_vec();

        // Toolbar: space toggle and gizmo mode
        ctx.horizontal(|h| {
            h.label("Space:");
            let options = ["Local", "World"];
            let mut space_idx = match self.space {
                CoordinateSpace::Local => 0,
                CoordinateSpace::World => 1,
            };
            if h.combo("##space", &mut space_idx, &options) {
                self.space = if space_idx == 0 {
                    CoordinateSpace::Local
                } else {
                    CoordinateSpace::World
                };
            }
        });

        ctx.horizontal(|h| {
            h.label("Gizmo:");
            if h.button(if self.gizmo_mode == GizmoMode::Translate {
                "[T]"
            } else {
                "T"
            }) {
                self.gizmo_mode = GizmoMode::Translate;
            }
            if h.button(if self.gizmo_mode == GizmoMode::Rotate {
                "[R]"
            } else {
                "R"
            }) {
                self.gizmo_mode = GizmoMode::Rotate;
            }
            if h.button(if self.gizmo_mode == GizmoMode::Scale {
                "[S]"
            } else {
                "S"
            }) {
                self.gizmo_mode = GizmoMode::Scale;
            }
        });

        ctx.separator();

        // Position (bytes 0-11)
        if let Some(pos) = TypeDecoder::decode_vec3(&data[0..12]) {
            let mut x = pos[0];
            let mut y = pos[1];
            let mut z = pos[2];

            ctx.collapsing("Position", |inner| {
                if inner.slider("X", &mut x, -1000.0, 1000.0) {
                    x = self.apply_snap(x, self.snap_position);
                    modified = true;
                }
                if inner.slider("Y", &mut y, -1000.0, 1000.0) {
                    y = self.apply_snap(y, self.snap_position);
                    modified = true;
                }
                if inner.slider("Z", &mut z, -1000.0, 1000.0) {
                    z = self.apply_snap(z, self.snap_position);
                    modified = true;
                }
            });

            if modified {
                result[0..4].copy_from_slice(&x.to_le_bytes());
                result[4..8].copy_from_slice(&y.to_le_bytes());
                result[8..12].copy_from_slice(&z.to_le_bytes());
            }
        }

        // Rotation (bytes 12-27)
        if let Some(rot) = TypeDecoder::decode_vec4(&data[12..28]) {
            let euler = self.quat_to_euler(rot);
            let mut ex = euler[0];
            let mut ey = euler[1];
            let mut ez = euler[2];

            ctx.collapsing("Rotation", |inner| {
                let mut euler_mode = self.show_euler;
                if inner.checkbox("Euler", &mut euler_mode) {
                    self.show_euler = euler_mode;
                }

                if self.show_euler {
                    if inner.slider("X (Roll)", &mut ex, -180.0, 180.0) {
                        ex = self.apply_snap(ex, self.snap_rotation);
                        modified = true;
                    }
                    if inner.slider("Y (Pitch)", &mut ey, -90.0, 90.0) {
                        ey = self.apply_snap(ey, self.snap_rotation);
                        modified = true;
                    }
                    if inner.slider("Z (Yaw)", &mut ez, -180.0, 180.0) {
                        ez = self.apply_snap(ez, self.snap_rotation);
                        modified = true;
                    }
                } else {
                    let mut qx = rot[0];
                    let mut qy = rot[1];
                    let mut qz = rot[2];
                    let mut qw = rot[3];

                    if inner.slider("X", &mut qx, -1.0, 1.0) {
                        modified = true;
                    }
                    if inner.slider("Y", &mut qy, -1.0, 1.0) {
                        modified = true;
                    }
                    if inner.slider("Z", &mut qz, -1.0, 1.0) {
                        modified = true;
                    }
                    if inner.slider("W", &mut qw, -1.0, 1.0) {
                        modified = true;
                    }

                    if modified {
                        // Normalize quaternion
                        let len = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
                        if len > f32::EPSILON {
                            qx /= len;
                            qy /= len;
                            qz /= len;
                            qw /= len;
                        }
                        result[12..16].copy_from_slice(&qx.to_le_bytes());
                        result[16..20].copy_from_slice(&qy.to_le_bytes());
                        result[20..24].copy_from_slice(&qz.to_le_bytes());
                        result[24..28].copy_from_slice(&qw.to_le_bytes());
                    }
                }
            });

            if modified && self.show_euler {
                let quat = self.euler_to_quat([ex, ey, ez]);
                result[12..16].copy_from_slice(&quat[0].to_le_bytes());
                result[16..20].copy_from_slice(&quat[1].to_le_bytes());
                result[20..24].copy_from_slice(&quat[2].to_le_bytes());
                result[24..28].copy_from_slice(&quat[3].to_le_bytes());
            }
        }

        // Scale (bytes 28-39, optional)
        if data.len() >= 40 {
            if let Some(scale) = TypeDecoder::decode_vec3(&data[28..40]) {
                let mut sx = scale[0];
                let mut sy = scale[1];
                let mut sz = scale[2];

                ctx.collapsing("Scale", |inner| {
                    let mut uniform =
                        (sx - sy).abs() < f32::EPSILON && (sy - sz).abs() < f32::EPSILON;

                    if inner.checkbox("Uniform", &mut uniform) {
                        if uniform {
                            sy = sx;
                            sz = sx;
                            modified = true;
                        }
                    }

                    if uniform {
                        if inner.slider("Scale", &mut sx, 0.001, 100.0) {
                            sx = self.apply_snap(sx, self.snap_scale);
                            sy = sx;
                            sz = sx;
                            modified = true;
                        }
                    } else {
                        if inner.slider("X", &mut sx, 0.001, 100.0) {
                            sx = self.apply_snap(sx, self.snap_scale);
                            modified = true;
                        }
                        if inner.slider("Y", &mut sy, 0.001, 100.0) {
                            sy = self.apply_snap(sy, self.snap_scale);
                            modified = true;
                        }
                        if inner.slider("Z", &mut sz, 0.001, 100.0) {
                            sz = self.apply_snap(sz, self.snap_scale);
                            modified = true;
                        }
                    }
                });

                if modified {
                    result[28..32].copy_from_slice(&sx.to_le_bytes());
                    result[32..36].copy_from_slice(&sy.to_le_bytes());
                    result[36..40].copy_from_slice(&sz.to_le_bytes());
                }
            }
        }

        // Snap settings
        ctx.collapsing("Snap Settings", |inner| {
            inner.checkbox("Enable Snap", &mut self.snap_enabled);
            inner.slider("Position", &mut self.snap_position, 0.01, 10.0);
            inner.slider("Rotation", &mut self.snap_rotation, 1.0, 90.0);
            inner.slider("Scale", &mut self.snap_scale, 0.01, 1.0);
        });

        if modified {
            Some(result)
        } else {
            None
        }
    }
}

impl Default for TransformView {
    fn default() -> Self {
        Self::new()
    }
}

impl MoldableView for TransformView {
    fn can_render(&self, component_name: &str, data: &[u8]) -> bool {
        let name_lower = component_name.to_lowercase();
        // Transform: 12 (pos) + 16 (quat) + 12 (scale) = 40 bytes minimum
        (name_lower.contains("transform") && data.len() >= 40)
            || (name_lower == "transform" && data.len() >= 28) // pos + quat only
    }

    fn render_mock(
        &mut self,
        ctx: &mut crate::egui_adapter::MockUIContext,
        component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        self.render(ctx, component_name, data)
    }

    fn name(&self) -> &str {
        "Transform"
    }

    fn priority(&self) -> i32 {
        100
    }
}

// ---------------------------------------------------------------------------
// Material View
// ---------------------------------------------------------------------------

/// PBR workflow type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PbrWorkflow {
    /// Metallic-roughness workflow.
    #[default]
    MetallicRoughness,
    /// Specular-glossiness workflow.
    SpecularGlossiness,
}

/// Specialized view for Material components.
///
/// Renders PBR material properties:
/// - Base color with color picker
/// - Metallic/roughness sliders
/// - Normal map intensity
/// - Emissive color and intensity
/// - Texture preview placeholders
pub struct MaterialView {
    /// Current PBR workflow.
    workflow: PbrWorkflow,
    /// Show advanced properties.
    show_advanced: bool,
}

impl MaterialView {
    /// Create a new material view.
    pub fn new() -> Self {
        Self {
            workflow: PbrWorkflow::MetallicRoughness,
            show_advanced: false,
        }
    }

    /// Set the PBR workflow.
    pub fn set_workflow(&mut self, workflow: PbrWorkflow) {
        self.workflow = workflow;
    }

    /// Get the current workflow.
    pub fn workflow(&self) -> PbrWorkflow {
        self.workflow
    }

    /// Render material component with any UIContext.
    pub fn render<T: UIContext>(
        &mut self,
        ctx: &mut T,
        _component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        if data.len() < 24 {
            ctx.label("Invalid Material data");
            return None;
        }

        let mut modified = false;
        let mut result = data.to_vec();

        // Workflow selector
        ctx.horizontal(|h| {
            h.label("Workflow:");
            let options = ["Metallic-Roughness", "Specular-Glossiness"];
            let mut idx = match self.workflow {
                PbrWorkflow::MetallicRoughness => 0,
                PbrWorkflow::SpecularGlossiness => 1,
            };
            if h.combo("##workflow", &mut idx, &options) {
                self.workflow = if idx == 0 {
                    PbrWorkflow::MetallicRoughness
                } else {
                    PbrWorkflow::SpecularGlossiness
                };
            }
        });

        ctx.separator();

        // Base color (bytes 0-15)
        if let Some(color) = TypeDecoder::decode_vec4(&data[0..16]) {
            let mut base_color = color;

            ctx.collapsing("Base Color", |inner| {
                if inner.color_edit("Color", &mut base_color) {
                    modified = true;
                }

                // Show hex value
                let r = (base_color[0] * 255.0) as u8;
                let g = (base_color[1] * 255.0) as u8;
                let b = (base_color[2] * 255.0) as u8;
                inner.label(&format!("#{:02X}{:02X}{:02X}", r, g, b));
            });

            if modified {
                for (i, &val) in base_color.iter().enumerate() {
                    let offset = i * 4;
                    result[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }

        // Metallic (bytes 16-19)
        if let Some(metallic) = TypeDecoder::decode_f32(&data[16..20]) {
            let mut m = metallic;
            ctx.horizontal(|h| {
                h.label("Metallic:");
                if h.slider("##metallic", &mut m, 0.0, 1.0) {
                    modified = true;
                    result[16..20].copy_from_slice(&m.to_le_bytes());
                }
            });
        }

        // Roughness (bytes 20-23)
        if let Some(roughness) = TypeDecoder::decode_f32(&data[20..24]) {
            let mut r = roughness;
            ctx.horizontal(|h| {
                h.label("Roughness:");
                if h.slider("##roughness", &mut r, 0.0, 1.0) {
                    modified = true;
                    result[20..24].copy_from_slice(&r.to_le_bytes());
                }
            });
        }

        // Emissive (bytes 24-39, optional)
        if data.len() >= 40 {
            if let Some(emissive) = TypeDecoder::decode_vec4(&data[24..40]) {
                let mut em = emissive;

                ctx.collapsing("Emissive", |inner| {
                    if inner.color_edit("Color", &mut em) {
                        modified = true;
                    }

                    // Emissive intensity (use alpha as intensity)
                    let mut intensity = em[3];
                    if inner.slider("Intensity", &mut intensity, 0.0, 10.0) {
                        em[3] = intensity;
                        modified = true;
                    }

                    if modified {
                        for (i, &val) in em.iter().enumerate() {
                            let offset = 24 + i * 4;
                            result[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                });
            }
        }

        // Advanced properties
        ctx.collapsing("Advanced", |inner| {
            inner.checkbox("Show Advanced", &mut self.show_advanced);

            if self.show_advanced && data.len() >= 48 {
                // Normal map intensity (bytes 40-43)
                if let Some(normal_intensity) = TypeDecoder::decode_f32(&data[40..44]) {
                    let mut ni = normal_intensity;
                    if inner.slider("Normal Intensity", &mut ni, 0.0, 2.0) {
                        modified = true;
                        result[40..44].copy_from_slice(&ni.to_le_bytes());
                    }
                }

                // AO intensity (bytes 44-47)
                if let Some(ao) = TypeDecoder::decode_f32(&data[44..48]) {
                    let mut ao_val = ao;
                    if inner.slider("AO Intensity", &mut ao_val, 0.0, 1.0) {
                        modified = true;
                        result[44..48].copy_from_slice(&ao_val.to_le_bytes());
                    }
                }
            }
        });

        // Texture slots (display only)
        ctx.collapsing("Textures", |inner| {
            inner.label("Albedo: (none)");
            inner.label("Normal: (none)");
            inner.label("Metallic/Roughness: (none)");
            inner.label("Emissive: (none)");
            inner.label("AO: (none)");
        });

        if modified {
            Some(result)
        } else {
            None
        }
    }
}

impl Default for MaterialView {
    fn default() -> Self {
        Self::new()
    }
}

impl MoldableView for MaterialView {
    fn can_render(&self, component_name: &str, data: &[u8]) -> bool {
        let name_lower = component_name.to_lowercase();
        // Material: base_color (16) + metallic (4) + roughness (4) + emissive (16) = 40 bytes min
        (name_lower.contains("material") || name_lower.contains("pbr")) && data.len() >= 24
    }

    fn render_mock(
        &mut self,
        ctx: &mut crate::egui_adapter::MockUIContext,
        component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        self.render(ctx, component_name, data)
    }

    fn name(&self) -> &str {
        "Material"
    }

    fn priority(&self) -> i32 {
        100
    }
}

// ---------------------------------------------------------------------------
// Mesh View
// ---------------------------------------------------------------------------

/// Specialized view for Mesh components.
///
/// Displays mesh statistics:
/// - Vertex count
/// - Triangle count
/// - Bounds (AABB)
/// - LOD information
/// - Mesh flags
pub struct MeshView {
    /// Show bounds visualization.
    show_bounds: bool,
    /// Show wireframe overlay.
    show_wireframe: bool,
}

impl MeshView {
    /// Create a new mesh view.
    pub fn new() -> Self {
        Self {
            show_bounds: false,
            show_wireframe: false,
        }
    }

    /// Toggle bounds visualization.
    pub fn set_show_bounds(&mut self, show: bool) {
        self.show_bounds = show;
    }

    /// Toggle wireframe overlay.
    pub fn set_show_wireframe(&mut self, show: bool) {
        self.show_wireframe = show;
    }

    /// Check if bounds are visible.
    pub fn show_bounds(&self) -> bool {
        self.show_bounds
    }

    /// Check if wireframe is visible.
    pub fn show_wireframe(&self) -> bool {
        self.show_wireframe
    }

    /// Render mesh component with any UIContext.
    pub fn render<T: UIContext>(
        &mut self,
        ctx: &mut T,
        _component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        if data.len() < 8 {
            ctx.label("Invalid Mesh data");
            return None;
        }

        // Mesh data is read-only, no modifications
        let modified = false;

        // Vertex count (bytes 0-3)
        let vertex_count = TypeDecoder::decode_u32(&data[0..4]).unwrap_or(0);

        // Triangle count (bytes 4-7)
        let triangle_count = TypeDecoder::decode_u32(&data[4..8]).unwrap_or(0);

        // Statistics
        ctx.collapsing("Statistics", |inner| {
            inner.label(&format!("Vertices: {}", vertex_count));
            inner.label(&format!("Triangles: {}", triangle_count));
            inner.label(&format!("Indices: {}", triangle_count * 3));

            // Estimated memory
            let vertex_size = 32; // pos(12) + normal(12) + uv(8)
            let index_size = 4; // u32
            let vertex_mem = vertex_count as usize * vertex_size;
            let index_mem = (triangle_count * 3) as usize * index_size;
            let total_mem = vertex_mem + index_mem;

            inner.separator();
            inner.label(&format!("Vertex data: {} KB", vertex_mem / 1024));
            inner.label(&format!("Index data: {} KB", index_mem / 1024));
            inner.label(&format!("Total: {} KB", total_mem / 1024));
        });

        // Bounds (bytes 8-31, optional)
        if data.len() >= 32 {
            let bounds_min = TypeDecoder::decode_vec3(&data[8..20]);
            let bounds_max = TypeDecoder::decode_vec3(&data[20..32]);

            if let (Some(min), Some(max)) = (bounds_min, bounds_max) {
                ctx.collapsing("Bounds", |inner| {
                    inner.checkbox("Show Bounds", &mut self.show_bounds);
                    inner.separator();

                    inner.label(&format!(
                        "Min: ({:.2}, {:.2}, {:.2})",
                        min[0], min[1], min[2]
                    ));
                    inner.label(&format!(
                        "Max: ({:.2}, {:.2}, {:.2})",
                        max[0], max[1], max[2]
                    ));

                    let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
                    inner.label(&format!(
                        "Size: ({:.2}, {:.2}, {:.2})",
                        size[0], size[1], size[2]
                    ));

                    let center = [
                        (min[0] + max[0]) * 0.5,
                        (min[1] + max[1]) * 0.5,
                        (min[2] + max[2]) * 0.5,
                    ];
                    inner.label(&format!(
                        "Center: ({:.2}, {:.2}, {:.2})",
                        center[0], center[1], center[2]
                    ));
                });
            }
        }

        // LOD info (bytes 32-35, optional)
        if data.len() >= 36 {
            let lod_count = TypeDecoder::decode_u32(&data[32..36]).unwrap_or(0);

            ctx.collapsing("LOD", |inner| {
                inner.label(&format!("LOD Levels: {}", lod_count));

                if lod_count > 0 {
                    // Display LOD distances if available
                    for i in 0..lod_count.min(4) {
                        let offset = 36 + (i as usize) * 4;
                        if data.len() >= offset + 4 {
                            if let Some(dist) = TypeDecoder::decode_f32(&data[offset..offset + 4]) {
                                inner.label(&format!("  LOD {}: {:.1}m", i, dist));
                            }
                        }
                    }
                }
            });
        }

        // Visualization toggles
        ctx.collapsing("Visualization", |inner| {
            inner.checkbox("Show Bounds", &mut self.show_bounds);
            inner.checkbox("Show Wireframe", &mut self.show_wireframe);
        });

        if modified {
            Some(data.to_vec())
        } else {
            None
        }
    }
}

impl Default for MeshView {
    fn default() -> Self {
        Self::new()
    }
}

impl MoldableView for MeshView {
    fn can_render(&self, component_name: &str, data: &[u8]) -> bool {
        let name_lower = component_name.to_lowercase();
        // Mesh: vertex_count (4) + triangle_count (4) + bounds_min (12) + bounds_max (12) = 32 bytes min
        (name_lower.contains("mesh") || name_lower.contains("geometry")) && data.len() >= 8
    }

    fn render_mock(
        &mut self,
        ctx: &mut crate::egui_adapter::MockUIContext,
        component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        self.render(ctx, component_name, data)
    }

    fn name(&self) -> &str {
        "Mesh"
    }

    fn priority(&self) -> i32 {
        100
    }
}

// ---------------------------------------------------------------------------
// Camera View
// ---------------------------------------------------------------------------

/// Camera projection type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ProjectionType {
    /// Perspective projection.
    #[default]
    Perspective,
    /// Orthographic projection.
    Orthographic,
}

/// Specialized view for Camera components.
///
/// Renders camera properties:
/// - FOV slider (perspective)
/// - Orthographic size
/// - Near/far planes
/// - Projection type toggle
/// - Frustum preview toggle
pub struct CameraView {
    /// Show frustum in viewport.
    show_frustum: bool,
    /// Preview camera view.
    preview_enabled: bool,
}

impl CameraView {
    /// Create a new camera view.
    pub fn new() -> Self {
        Self {
            show_frustum: false,
            preview_enabled: false,
        }
    }

    /// Set frustum visibility.
    pub fn set_show_frustum(&mut self, show: bool) {
        self.show_frustum = show;
    }

    /// Set preview enabled.
    pub fn set_preview_enabled(&mut self, enabled: bool) {
        self.preview_enabled = enabled;
    }

    /// Check if frustum is visible.
    pub fn show_frustum(&self) -> bool {
        self.show_frustum
    }

    /// Check if preview is enabled.
    pub fn preview_enabled(&self) -> bool {
        self.preview_enabled
    }

    /// Render camera component with any UIContext.
    pub fn render<T: UIContext>(
        &mut self,
        ctx: &mut T,
        _component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        if data.len() < 13 {
            ctx.label("Invalid Camera data");
            return None;
        }

        let mut modified = false;
        let mut result = data.to_vec();

        // Projection type (byte 0)
        let proj_type = if data[0] == 0 {
            ProjectionType::Perspective
        } else {
            ProjectionType::Orthographic
        };

        ctx.horizontal(|h| {
            h.label("Projection:");
            let options = ["Perspective", "Orthographic"];
            let mut idx = match proj_type {
                ProjectionType::Perspective => 0,
                ProjectionType::Orthographic => 1,
            };
            if h.combo("##proj", &mut idx, &options) {
                result[0] = idx as u8;
                modified = true;
            }
        });

        ctx.separator();

        // FOV or ortho size (bytes 1-4)
        if let Some(fov_or_size) = TypeDecoder::decode_f32(&data[1..5]) {
            let mut value = fov_or_size;

            if proj_type == ProjectionType::Perspective {
                ctx.horizontal(|h| {
                    h.label("Field of View:");
                    if h.slider("##fov", &mut value, 10.0, 120.0) {
                        modified = true;
                        result[1..5].copy_from_slice(&value.to_le_bytes());
                    }
                });
                ctx.label(&format!("({:.1} degrees)", value));
            } else {
                ctx.horizontal(|h| {
                    h.label("Ortho Size:");
                    if h.slider("##size", &mut value, 0.1, 100.0) {
                        modified = true;
                        result[1..5].copy_from_slice(&value.to_le_bytes());
                    }
                });
            }
        }

        // Near plane (bytes 5-8)
        if let Some(near) = TypeDecoder::decode_f32(&data[5..9]) {
            let mut n = near;
            ctx.horizontal(|h| {
                h.label("Near Plane:");
                if h.slider("##near", &mut n, 0.001, 10.0) {
                    modified = true;
                    result[5..9].copy_from_slice(&n.to_le_bytes());
                }
            });
        }

        // Far plane (bytes 9-12)
        if let Some(far) = TypeDecoder::decode_f32(&data[9..13]) {
            let mut f = far;
            ctx.horizontal(|h| {
                h.label("Far Plane:");
                if h.slider("##far", &mut f, 10.0, 10000.0) {
                    modified = true;
                    result[9..13].copy_from_slice(&f.to_le_bytes());
                }
            });
        }

        // Aspect ratio (bytes 13-16, optional)
        if data.len() >= 17 {
            if let Some(aspect) = TypeDecoder::decode_f32(&data[13..17]) {
                ctx.label(&format!("Aspect Ratio: {:.3}", aspect));
            }
        }

        ctx.separator();

        // Visualization
        ctx.checkbox("Show Frustum", &mut self.show_frustum);
        ctx.checkbox("Preview", &mut self.preview_enabled);

        if modified {
            Some(result)
        } else {
            None
        }
    }
}

impl Default for CameraView {
    fn default() -> Self {
        Self::new()
    }
}

impl MoldableView for CameraView {
    fn can_render(&self, component_name: &str, data: &[u8]) -> bool {
        let name_lower = component_name.to_lowercase();
        // Camera: projection_type (1) + fov (4) + near (4) + far (4) + aspect (4) = 17 bytes min
        name_lower.contains("camera") && data.len() >= 13
    }

    fn render_mock(
        &mut self,
        ctx: &mut crate::egui_adapter::MockUIContext,
        component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        self.render(ctx, component_name, data)
    }

    fn name(&self) -> &str {
        "Camera"
    }

    fn priority(&self) -> i32 {
        100
    }
}

// ---------------------------------------------------------------------------
// Light View
// ---------------------------------------------------------------------------

/// Light type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LightType {
    /// Directional light (sun).
    #[default]
    Directional,
    /// Point light.
    Point,
    /// Spot light.
    Spot,
    /// Area light (rectangle).
    Area,
}

/// Specialized view for Light components.
///
/// Renders light properties:
/// - Light type selector
/// - Color picker
/// - Intensity slider
/// - Range (point/spot)
/// - Spot angle (spot)
/// - Shadow settings
pub struct LightView {
    /// Show light gizmo in viewport.
    show_gizmo: bool,
    /// Show shadow frustum.
    show_shadow_frustum: bool,
}

impl LightView {
    /// Create a new light view.
    pub fn new() -> Self {
        Self {
            show_gizmo: true,
            show_shadow_frustum: false,
        }
    }

    /// Set gizmo visibility.
    pub fn set_show_gizmo(&mut self, show: bool) {
        self.show_gizmo = show;
    }

    /// Set shadow frustum visibility.
    pub fn set_show_shadow_frustum(&mut self, show: bool) {
        self.show_shadow_frustum = show;
    }

    /// Check if gizmo is visible.
    pub fn show_gizmo(&self) -> bool {
        self.show_gizmo
    }

    /// Check if shadow frustum is visible.
    pub fn show_shadow_frustum(&self) -> bool {
        self.show_shadow_frustum
    }

    /// Approximate color temperature from RGB (very rough estimate).
    fn rgb_to_kelvin(r: f32, _g: f32, b: f32) -> f32 {
        // Rough approximation based on blue-red ratio
        if r < f32::EPSILON {
            return 10000.0;
        }
        let ratio = b / r;
        // Map ratio to temperature range
        2000.0 + ratio * 8000.0
    }

    /// Render light component with any UIContext.
    pub fn render<T: UIContext>(
        &mut self,
        ctx: &mut T,
        _component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        if data.len() < 21 {
            ctx.label("Invalid Light data");
            return None;
        }

        let mut modified = false;
        let mut result = data.to_vec();

        // Light type (byte 0)
        let light_type = match data[0] {
            0 => LightType::Directional,
            1 => LightType::Point,
            2 => LightType::Spot,
            3 => LightType::Area,
            _ => LightType::Point,
        };

        ctx.horizontal(|h| {
            h.label("Type:");
            let options = ["Directional", "Point", "Spot", "Area"];
            let mut idx = match light_type {
                LightType::Directional => 0,
                LightType::Point => 1,
                LightType::Spot => 2,
                LightType::Area => 3,
            };
            if h.combo("##type", &mut idx, &options) {
                result[0] = idx as u8;
                modified = true;
            }
        });

        ctx.separator();

        // Color (bytes 1-16)
        if let Some(color) = TypeDecoder::decode_vec4(&data[1..17]) {
            let mut c = color;

            ctx.collapsing("Color", |inner| {
                if inner.color_edit("Light Color", &mut c) {
                    modified = true;
                    for (i, &val) in c.iter().enumerate() {
                        let offset = 1 + i * 4;
                        result[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                    }
                }

                // Color temperature (approximate)
                let kelvin = Self::rgb_to_kelvin(c[0], c[1], c[2]);
                inner.label(&format!("~{:.0}K", kelvin));
            });
        }

        // Intensity (bytes 17-20)
        if let Some(intensity) = TypeDecoder::decode_f32(&data[17..21]) {
            let mut i = intensity;
            ctx.horizontal(|h| {
                h.label("Intensity:");
                if h.slider("##intensity", &mut i, 0.0, 100.0) {
                    modified = true;
                    result[17..21].copy_from_slice(&i.to_le_bytes());
                }
            });

            // Show lumens estimate
            let lumens = i * 1000.0; // Rough approximation
            ctx.label(&format!("~{:.0} lumens", lumens));
        }

        // Range (bytes 21-24, for point/spot)
        if data.len() >= 25 && light_type != LightType::Directional {
            if let Some(range) = TypeDecoder::decode_f32(&data[21..25]) {
                let mut r = range;
                ctx.horizontal(|h| {
                    h.label("Range:");
                    if h.slider("##range", &mut r, 0.1, 100.0) {
                        modified = true;
                        result[21..25].copy_from_slice(&r.to_le_bytes());
                    }
                });
            }
        }

        // Spot angle (bytes 25-28, for spot)
        if data.len() >= 29 && light_type == LightType::Spot {
            if let Some(angle) = TypeDecoder::decode_f32(&data[25..29]) {
                let mut a = angle;
                ctx.horizontal(|h| {
                    h.label("Spot Angle:");
                    if h.slider("##angle", &mut a, 1.0, 179.0) {
                        modified = true;
                        result[25..29].copy_from_slice(&a.to_le_bytes());
                    }
                });

                // Inner angle (bytes 29-32)
                if data.len() >= 33 {
                    if let Some(inner) = TypeDecoder::decode_f32(&data[29..33]) {
                        let mut inner_angle = inner;
                        ctx.horizontal(|h| {
                            h.label("Inner Angle:");
                            if h.slider("##inner", &mut inner_angle, 0.0, a) {
                                modified = true;
                                result[29..33].copy_from_slice(&inner_angle.to_le_bytes());
                            }
                        });
                    }
                }
            }
        }

        // Shadow settings
        ctx.collapsing("Shadows", |inner| {
            // Shadow enabled (byte after light-specific data)
            let shadow_offset = match light_type {
                LightType::Directional => 21,
                LightType::Point => 25,
                LightType::Spot => 33,
                LightType::Area => 25,
            };

            if data.len() > shadow_offset {
                let mut shadows_enabled = data[shadow_offset] != 0;
                if inner.checkbox("Cast Shadows", &mut shadows_enabled) {
                    modified = true;
                    result[shadow_offset] = if shadows_enabled { 1 } else { 0 };
                }

                if shadows_enabled {
                    // Shadow bias
                    if data.len() >= shadow_offset + 5 {
                        if let Some(bias) =
                            TypeDecoder::decode_f32(&data[shadow_offset + 1..shadow_offset + 5])
                        {
                            let mut b = bias;
                            if inner.slider("Bias", &mut b, 0.0, 0.1) {
                                modified = true;
                                result[shadow_offset + 1..shadow_offset + 5]
                                    .copy_from_slice(&b.to_le_bytes());
                            }
                        }
                    }

                    inner.checkbox("Show Shadow Frustum", &mut self.show_shadow_frustum);
                }
            }
        });

        // Visualization
        ctx.separator();
        ctx.checkbox("Show Gizmo", &mut self.show_gizmo);

        if modified {
            Some(result)
        } else {
            None
        }
    }
}

impl Default for LightView {
    fn default() -> Self {
        Self::new()
    }
}

impl MoldableView for LightView {
    fn can_render(&self, component_name: &str, data: &[u8]) -> bool {
        let name_lower = component_name.to_lowercase();
        // Light: type (1) + color (16) + intensity (4) + range (4) = 25 bytes min
        name_lower.contains("light") && data.len() >= 21
    }

    fn render_mock(
        &mut self,
        ctx: &mut crate::egui_adapter::MockUIContext,
        component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        self.render(ctx, component_name, data)
    }

    fn name(&self) -> &str {
        "Light"
    }

    fn priority(&self) -> i32 {
        100
    }
}

// ---------------------------------------------------------------------------
// Physics View
// ---------------------------------------------------------------------------

/// Collision shape type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CollisionShapeType {
    /// Box collider.
    #[default]
    Box,
    /// Sphere collider.
    Sphere,
    /// Capsule collider.
    Capsule,
    /// Mesh collider.
    Mesh,
    /// Convex hull collider.
    ConvexHull,
}

/// Specialized view for Physics/RigidBody components.
///
/// Renders physics properties:
/// - Mass
/// - Velocity vectors
/// - Angular velocity
/// - Collision shape type
/// - Physics material (friction, restitution)
pub struct PhysicsView {
    /// Show velocity vectors in viewport.
    show_velocity: bool,
    /// Show collision shape in viewport.
    show_collider: bool,
}

impl PhysicsView {
    /// Create a new physics view.
    pub fn new() -> Self {
        Self {
            show_velocity: false,
            show_collider: true,
        }
    }

    /// Set velocity visibility.
    pub fn set_show_velocity(&mut self, show: bool) {
        self.show_velocity = show;
    }

    /// Set collider visibility.
    pub fn set_show_collider(&mut self, show: bool) {
        self.show_collider = show;
    }

    /// Check if velocity is visible.
    pub fn show_velocity(&self) -> bool {
        self.show_velocity
    }

    /// Check if collider is visible.
    pub fn show_collider(&self) -> bool {
        self.show_collider
    }

    /// Render physics component with any UIContext.
    pub fn render<T: UIContext>(
        &mut self,
        ctx: &mut T,
        _component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        if data.len() < 4 {
            ctx.label("Invalid Physics data");
            return None;
        }

        let mut modified = false;
        let mut result = data.to_vec();

        // Mass (bytes 0-3)
        if let Some(mass) = TypeDecoder::decode_f32(&data[0..4]) {
            let mut m = mass;
            ctx.horizontal(|h| {
                h.label("Mass:");
                if h.slider("##mass", &mut m, 0.0, 1000.0) {
                    modified = true;
                    result[0..4].copy_from_slice(&m.to_le_bytes());
                }
            });

            // Inverse mass for info
            if m > f32::EPSILON {
                ctx.label(&format!("Inverse Mass: {:.4}", 1.0 / m));
            } else {
                ctx.label("Static (infinite mass)");
            }
        }

        ctx.separator();

        // Velocity (bytes 4-15)
        if data.len() >= 16 {
            if let Some(vel) = TypeDecoder::decode_vec3(&data[4..16]) {
                let mut vx = vel[0];
                let mut vy = vel[1];
                let mut vz = vel[2];

                ctx.collapsing("Linear Velocity", |inner| {
                    if inner.slider("X", &mut vx, -100.0, 100.0) {
                        modified = true;
                    }
                    if inner.slider("Y", &mut vy, -100.0, 100.0) {
                        modified = true;
                    }
                    if inner.slider("Z", &mut vz, -100.0, 100.0) {
                        modified = true;
                    }

                    let speed = (vx * vx + vy * vy + vz * vz).sqrt();
                    inner.label(&format!("Speed: {:.2} m/s", speed));

                    if modified {
                        result[4..8].copy_from_slice(&vx.to_le_bytes());
                        result[8..12].copy_from_slice(&vy.to_le_bytes());
                        result[12..16].copy_from_slice(&vz.to_le_bytes());
                    }
                });
            }
        }

        // Angular velocity (bytes 16-27)
        if data.len() >= 28 {
            if let Some(ang_vel) = TypeDecoder::decode_vec3(&data[16..28]) {
                let mut wx = ang_vel[0];
                let mut wy = ang_vel[1];
                let mut wz = ang_vel[2];

                ctx.collapsing("Angular Velocity", |inner| {
                    if inner.slider("X", &mut wx, -10.0, 10.0) {
                        modified = true;
                    }
                    if inner.slider("Y", &mut wy, -10.0, 10.0) {
                        modified = true;
                    }
                    if inner.slider("Z", &mut wz, -10.0, 10.0) {
                        modified = true;
                    }

                    let angular_speed = (wx * wx + wy * wy + wz * wz).sqrt();
                    inner.label(&format!("Angular Speed: {:.2} rad/s", angular_speed));

                    if modified {
                        result[16..20].copy_from_slice(&wx.to_le_bytes());
                        result[20..24].copy_from_slice(&wy.to_le_bytes());
                        result[24..28].copy_from_slice(&wz.to_le_bytes());
                    }
                });
            }
        }

        // Collision shape type (byte 28)
        if data.len() >= 29 {
            let shape_type = match data[28] {
                0 => CollisionShapeType::Box,
                1 => CollisionShapeType::Sphere,
                2 => CollisionShapeType::Capsule,
                3 => CollisionShapeType::Mesh,
                4 => CollisionShapeType::ConvexHull,
                _ => CollisionShapeType::Box,
            };

            ctx.horizontal(|h| {
                h.label("Shape:");
                let options = ["Box", "Sphere", "Capsule", "Mesh", "ConvexHull"];
                let mut idx = match shape_type {
                    CollisionShapeType::Box => 0,
                    CollisionShapeType::Sphere => 1,
                    CollisionShapeType::Capsule => 2,
                    CollisionShapeType::Mesh => 3,
                    CollisionShapeType::ConvexHull => 4,
                };
                if h.combo("##shape", &mut idx, &options) {
                    result[28] = idx as u8;
                    modified = true;
                }
            });
        }

        // Physics material (friction, restitution)
        if data.len() >= 37 {
            ctx.collapsing("Material", |inner| {
                // Friction (bytes 29-32)
                if let Some(friction) = TypeDecoder::decode_f32(&data[29..33]) {
                    let mut f = friction;
                    if inner.slider("Friction", &mut f, 0.0, 1.0) {
                        modified = true;
                        result[29..33].copy_from_slice(&f.to_le_bytes());
                    }
                }

                // Restitution (bytes 33-36)
                if let Some(restitution) = TypeDecoder::decode_f32(&data[33..37]) {
                    let mut r = restitution;
                    if inner.slider("Restitution", &mut r, 0.0, 1.0) {
                        modified = true;
                        result[33..37].copy_from_slice(&r.to_le_bytes());
                    }
                }
            });
        }

        // Visualization
        ctx.separator();
        ctx.checkbox("Show Velocity", &mut self.show_velocity);
        ctx.checkbox("Show Collider", &mut self.show_collider);

        if modified {
            Some(result)
        } else {
            None
        }
    }
}

impl Default for PhysicsView {
    fn default() -> Self {
        Self::new()
    }
}

impl MoldableView for PhysicsView {
    fn can_render(&self, component_name: &str, data: &[u8]) -> bool {
        let name_lower = component_name.to_lowercase();
        // Physics: mass (4) + velocity (12) + angular_velocity (12) + shape_type (1) = 29 bytes min
        (name_lower.contains("physics")
            || name_lower.contains("rigidbody")
            || name_lower.contains("rigid_body")
            || name_lower.contains("collider"))
            && data.len() >= 4
    }

    fn render_mock(
        &mut self,
        ctx: &mut crate::egui_adapter::MockUIContext,
        component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        self.render(ctx, component_name, data)
    }

    fn name(&self) -> &str {
        "Physics"
    }

    fn priority(&self) -> i32 {
        100
    }
}

// ---------------------------------------------------------------------------
// Moldable Registry
// ---------------------------------------------------------------------------

/// Registry of moldable views for component visualization.
///
/// The registry maintains a list of views and finds the best match
/// for a given component type.
pub struct MoldableRegistry {
    /// Transform view.
    pub transform: TransformView,
    /// Material view.
    pub material: MaterialView,
    /// Mesh view.
    pub mesh: MeshView,
    /// Camera view.
    pub camera: CameraView,
    /// Light view.
    pub light: LightView,
    /// Physics view.
    pub physics: PhysicsView,
}

impl MoldableRegistry {
    /// Create a new registry with all standard views.
    pub fn new() -> Self {
        Self {
            transform: TransformView::new(),
            material: MaterialView::new(),
            mesh: MeshView::new(),
            camera: CameraView::new(),
            light: LightView::new(),
            physics: PhysicsView::new(),
        }
    }

    /// Find the name of the best view for a component.
    pub fn find_view_name(&self, component_name: &str, data: &[u8]) -> Option<&'static str> {
        // Check in order of priority
        if self.transform.can_render(component_name, data) {
            return Some("Transform");
        }
        if self.material.can_render(component_name, data) {
            return Some("Material");
        }
        if self.camera.can_render(component_name, data) {
            return Some("Camera");
        }
        if self.light.can_render(component_name, data) {
            return Some("Light");
        }
        if self.physics.can_render(component_name, data) {
            return Some("Physics");
        }
        if self.mesh.can_render(component_name, data) {
            return Some("Mesh");
        }
        None
    }

    /// Check if a view for the given component exists.
    pub fn has_view_for(&self, component_name: &str, data: &[u8]) -> bool {
        self.find_view_name(component_name, data).is_some()
    }

    /// Render a component with the best matching view.
    ///
    /// Returns `Some(modified_bytes)` if the component was modified.
    pub fn render<T: UIContext>(
        &mut self,
        ctx: &mut T,
        component_name: &str,
        data: &[u8],
    ) -> Option<Vec<u8>> {
        // Try specialized views in order
        if self.transform.can_render(component_name, data) {
            return self.transform.render(ctx, component_name, data);
        }
        if self.material.can_render(component_name, data) {
            return self.material.render(ctx, component_name, data);
        }
        if self.camera.can_render(component_name, data) {
            return self.camera.render(ctx, component_name, data);
        }
        if self.light.can_render(component_name, data) {
            return self.light.render(ctx, component_name, data);
        }
        if self.physics.can_render(component_name, data) {
            return self.physics.render(ctx, component_name, data);
        }
        if self.mesh.can_render(component_name, data) {
            return self.mesh.render(ctx, component_name, data);
        }

        // Fallback: raw bytes display
        ctx.label(&format!("{}: {} bytes", component_name, data.len()));
        if data.len() <= 64 {
            let hex: Vec<String> = data.iter().map(|b| format!("{:02X}", b)).collect();
            ctx.label(&hex.join(" "));
        } else {
            ctx.label("(data too large for raw display)");
        }
        None
    }

    /// Get the number of registered views (always 6 for standard views).
    pub fn view_count(&self) -> usize {
        6
    }

    /// Get all registered view names.
    pub fn view_names(&self) -> Vec<&'static str> {
        vec![
            "Transform",
            "Material",
            "Mesh",
            "Camera",
            "Light",
            "Physics",
        ]
    }
}

impl Default for MoldableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egui_adapter::MockUIContext;

    // -------------------------------------------------------------------------
    // TransformView Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transform_view_new() {
        let view = TransformView::new();
        assert_eq!(view.space(), CoordinateSpace::Local);
        assert_eq!(view.gizmo_mode(), GizmoMode::Translate);
        assert!(view.show_euler());
    }

    #[test]
    fn test_transform_view_can_render() {
        let view = TransformView::new();
        let data = vec![0u8; 40];
        assert!(view.can_render("Transform", &data));
        assert!(view.can_render("LocalTransform", &data));
        assert!(view.can_render("TRANSFORM", &data));
        assert!(!view.can_render("Transform", &[0u8; 20]));
        assert!(!view.can_render("Position", &data));
    }

    #[test]
    fn test_transform_view_name() {
        let view = TransformView::new();
        assert_eq!(view.name(), "Transform");
    }

    #[test]
    fn test_transform_view_priority() {
        let view = TransformView::new();
        assert_eq!(view.priority(), 100);
    }

    #[test]
    fn test_transform_view_space_toggle() {
        let mut view = TransformView::new();
        view.set_space(CoordinateSpace::World);
        assert_eq!(view.space(), CoordinateSpace::World);
    }

    #[test]
    fn test_transform_view_gizmo_mode() {
        let mut view = TransformView::new();
        view.set_gizmo_mode(GizmoMode::Rotate);
        assert_eq!(view.gizmo_mode(), GizmoMode::Rotate);
    }

    #[test]
    fn test_transform_view_euler_toggle() {
        let mut view = TransformView::new();
        view.set_show_euler(false);
        assert!(!view.show_euler());
    }

    #[test]
    fn test_transform_view_render() {
        let mut view = TransformView::new();
        let mut ctx = MockUIContext::new(1);

        // Create transform data: pos(12) + rot(16) + scale(12) = 40 bytes
        let mut data = Vec::with_capacity(40);
        // Position
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        // Rotation (identity quaternion)
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        // Scale
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());

        let result = view.render(&mut ctx, "Transform", &data);
        // No modification without interaction
        assert!(result.is_none());
    }

    #[test]
    fn test_transform_view_quat_euler_conversion() {
        let view = TransformView::new();

        // Identity quaternion should give zero euler angles
        let quat = [0.0, 0.0, 0.0, 1.0];
        let euler = view.quat_to_euler(quat);
        assert!(euler[0].abs() < 0.001);
        assert!(euler[1].abs() < 0.001);
        assert!(euler[2].abs() < 0.001);

        // Round trip
        let quat2 = view.euler_to_quat(euler);
        assert!((quat2[0] - quat[0]).abs() < 0.001);
        assert!((quat2[1] - quat[1]).abs() < 0.001);
        assert!((quat2[2] - quat[2]).abs() < 0.001);
        assert!((quat2[3] - quat[3]).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // MaterialView Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_material_view_new() {
        let view = MaterialView::new();
        assert_eq!(view.workflow(), PbrWorkflow::MetallicRoughness);
    }

    #[test]
    fn test_material_view_can_render() {
        let view = MaterialView::new();
        let data = vec![0u8; 40];
        assert!(view.can_render("Material", &data));
        assert!(view.can_render("PBRMaterial", &data));
        assert!(view.can_render("pbr_material", &data));
        assert!(!view.can_render("Material", &[0u8; 10]));
    }

    #[test]
    fn test_material_view_name() {
        let view = MaterialView::new();
        assert_eq!(view.name(), "Material");
    }

    #[test]
    fn test_material_view_workflow() {
        let mut view = MaterialView::new();
        view.set_workflow(PbrWorkflow::SpecularGlossiness);
        assert_eq!(view.workflow(), PbrWorkflow::SpecularGlossiness);
    }

    #[test]
    fn test_material_view_render() {
        let mut view = MaterialView::new();
        let mut ctx = MockUIContext::new(1);

        // Create material data: base_color(16) + metallic(4) + roughness(4) = 24 bytes
        let mut data = Vec::with_capacity(24);
        // Base color (white)
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        // Metallic
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // Roughness
        data.extend_from_slice(&0.5f32.to_le_bytes());

        let result = view.render(&mut ctx, "Material", &data);
        assert!(result.is_none());
    }

    // -------------------------------------------------------------------------
    // MeshView Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mesh_view_new() {
        let view = MeshView::new();
        assert!(!view.show_bounds());
        assert!(!view.show_wireframe());
    }

    #[test]
    fn test_mesh_view_can_render() {
        let view = MeshView::new();
        let data = vec![0u8; 32];
        assert!(view.can_render("Mesh", &data));
        assert!(view.can_render("StaticMesh", &data));
        assert!(view.can_render("geometry", &data));
        assert!(!view.can_render("Mesh", &[0u8; 4]));
    }

    #[test]
    fn test_mesh_view_name() {
        let view = MeshView::new();
        assert_eq!(view.name(), "Mesh");
    }

    #[test]
    fn test_mesh_view_toggles() {
        let mut view = MeshView::new();
        view.set_show_bounds(true);
        view.set_show_wireframe(true);
        assert!(view.show_bounds());
        assert!(view.show_wireframe());
    }

    #[test]
    fn test_mesh_view_render() {
        let mut view = MeshView::new();
        let mut ctx = MockUIContext::new(1);

        // Create mesh data: vertex_count(4) + triangle_count(4)
        let mut data = Vec::with_capacity(8);
        data.extend_from_slice(&1000u32.to_le_bytes()); // vertices
        data.extend_from_slice(&500u32.to_le_bytes()); // triangles

        let result = view.render(&mut ctx, "Mesh", &data);
        // Mesh view is read-only
        assert!(result.is_none());
    }

    // -------------------------------------------------------------------------
    // CameraView Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_camera_view_new() {
        let view = CameraView::new();
        assert!(!view.show_frustum());
        assert!(!view.preview_enabled());
    }

    #[test]
    fn test_camera_view_can_render() {
        let view = CameraView::new();
        let data = vec![0u8; 17];
        assert!(view.can_render("Camera", &data));
        assert!(view.can_render("MainCamera", &data));
        assert!(!view.can_render("Camera", &[0u8; 10]));
    }

    #[test]
    fn test_camera_view_name() {
        let view = CameraView::new();
        assert_eq!(view.name(), "Camera");
    }

    #[test]
    fn test_camera_view_toggles() {
        let mut view = CameraView::new();
        view.set_show_frustum(true);
        view.set_preview_enabled(true);
        assert!(view.show_frustum());
        assert!(view.preview_enabled());
    }

    #[test]
    fn test_camera_view_render() {
        let mut view = CameraView::new();
        let mut ctx = MockUIContext::new(1);

        // Create camera data: proj_type(1) + fov(4) + near(4) + far(4) = 13 bytes
        let mut data = vec![0u8]; // Perspective
        data.extend_from_slice(&60.0f32.to_le_bytes()); // FOV
        data.extend_from_slice(&0.1f32.to_le_bytes()); // Near
        data.extend_from_slice(&1000.0f32.to_le_bytes()); // Far

        let result = view.render(&mut ctx, "Camera", &data);
        assert!(result.is_none());
    }

    // -------------------------------------------------------------------------
    // LightView Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_light_view_new() {
        let view = LightView::new();
        assert!(view.show_gizmo());
        assert!(!view.show_shadow_frustum());
    }

    #[test]
    fn test_light_view_can_render() {
        let view = LightView::new();
        let data = vec![0u8; 25];
        assert!(view.can_render("Light", &data));
        assert!(view.can_render("PointLight", &data));
        assert!(view.can_render("DirectionalLight", &data));
        assert!(!view.can_render("Light", &[0u8; 10]));
    }

    #[test]
    fn test_light_view_name() {
        let view = LightView::new();
        assert_eq!(view.name(), "Light");
    }

    #[test]
    fn test_light_view_toggles() {
        let mut view = LightView::new();
        view.set_show_gizmo(false);
        view.set_show_shadow_frustum(true);
        assert!(!view.show_gizmo());
        assert!(view.show_shadow_frustum());
    }

    #[test]
    fn test_light_view_render() {
        let mut view = LightView::new();
        let mut ctx = MockUIContext::new(1);

        // Create light data: type(1) + color(16) + intensity(4) = 21 bytes
        let mut data = vec![1u8]; // Point light
        // Color (white)
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        // Intensity
        data.extend_from_slice(&10.0f32.to_le_bytes());

        let result = view.render(&mut ctx, "Light", &data);
        assert!(result.is_none());
    }

    #[test]
    fn test_light_view_rgb_to_kelvin() {
        // Pure red should be warm
        let kelvin = LightView::rgb_to_kelvin(1.0, 0.0, 0.0);
        assert!(kelvin < 4000.0);

        // Pure blue should be cool
        let kelvin = LightView::rgb_to_kelvin(0.0, 0.0, 1.0);
        assert!(kelvin > 8000.0);
    }

    // -------------------------------------------------------------------------
    // PhysicsView Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_physics_view_new() {
        let view = PhysicsView::new();
        assert!(!view.show_velocity());
        assert!(view.show_collider());
    }

    #[test]
    fn test_physics_view_can_render() {
        let view = PhysicsView::new();
        let data = vec![0u8; 29];
        assert!(view.can_render("Physics", &data));
        assert!(view.can_render("RigidBody", &data));
        assert!(view.can_render("rigid_body", &data));
        assert!(view.can_render("Collider", &data));
        assert!(!view.can_render("Physics", &[0u8; 2]));
    }

    #[test]
    fn test_physics_view_name() {
        let view = PhysicsView::new();
        assert_eq!(view.name(), "Physics");
    }

    #[test]
    fn test_physics_view_toggles() {
        let mut view = PhysicsView::new();
        view.set_show_velocity(true);
        view.set_show_collider(false);
        assert!(view.show_velocity());
        assert!(!view.show_collider());
    }

    #[test]
    fn test_physics_view_render() {
        let mut view = PhysicsView::new();
        let mut ctx = MockUIContext::new(1);

        // Create physics data: mass(4) = 4 bytes minimum
        let mut data = Vec::with_capacity(4);
        data.extend_from_slice(&10.0f32.to_le_bytes()); // Mass

        let result = view.render(&mut ctx, "Physics", &data);
        assert!(result.is_none());
    }

    // -------------------------------------------------------------------------
    // MoldableRegistry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_registry_new() {
        let registry = MoldableRegistry::new();
        assert_eq!(registry.view_count(), 6);
    }

    #[test]
    fn test_registry_view_names() {
        let registry = MoldableRegistry::new();
        let names = registry.view_names();
        assert!(names.contains(&"Transform"));
        assert!(names.contains(&"Material"));
        assert!(names.contains(&"Mesh"));
        assert!(names.contains(&"Camera"));
        assert!(names.contains(&"Light"));
        assert!(names.contains(&"Physics"));
    }

    #[test]
    fn test_registry_find_view_transform() {
        let registry = MoldableRegistry::new();
        let data = vec![0u8; 40];
        let view = registry.find_view_name("Transform", &data);
        assert!(view.is_some());
        assert_eq!(view.unwrap(), "Transform");
    }

    #[test]
    fn test_registry_find_view_material() {
        let registry = MoldableRegistry::new();
        let data = vec![0u8; 40];
        let view = registry.find_view_name("Material", &data);
        assert!(view.is_some());
        assert_eq!(view.unwrap(), "Material");
    }

    #[test]
    fn test_registry_find_view_none() {
        let registry = MoldableRegistry::new();
        let data = vec![0u8; 4];
        let view = registry.find_view_name("UnknownComponent", &data);
        assert!(view.is_none());
    }

    #[test]
    fn test_registry_has_view_for() {
        let registry = MoldableRegistry::new();
        assert!(registry.has_view_for("Transform", &vec![0u8; 40]));
        assert!(registry.has_view_for("Camera", &vec![0u8; 17]));
        assert!(!registry.has_view_for("Unknown", &vec![0u8; 10]));
    }

    #[test]
    fn test_registry_render_known() {
        let mut registry = MoldableRegistry::new();
        let mut ctx = MockUIContext::new(1);

        let mut data = Vec::with_capacity(40);
        for _ in 0..10 {
            data.extend_from_slice(&0.0f32.to_le_bytes());
        }

        let result = registry.render(&mut ctx, "Transform", &data);
        assert!(result.is_none()); // No modification
    }

    #[test]
    fn test_registry_render_unknown() {
        let mut registry = MoldableRegistry::new();
        let mut ctx = MockUIContext::new(1);

        let data = vec![0x01, 0x02, 0x03, 0x04];
        let result = registry.render(&mut ctx, "Unknown", &data);
        assert!(result.is_none());

        // Should have rendered raw bytes
        let ops = ctx.operations();
        assert!(ops.iter().any(
            |op| matches!(op, crate::egui_adapter::MockOperation::Label(s) if s.contains("Unknown"))
        ));
    }

    #[test]
    fn test_registry_default() {
        let registry: MoldableRegistry = Default::default();
        assert_eq!(registry.view_count(), 6);
    }

    // -------------------------------------------------------------------------
    // Coordinate Space and Gizmo Mode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coordinate_space_default() {
        let space: CoordinateSpace = Default::default();
        assert_eq!(space, CoordinateSpace::Local);
    }

    #[test]
    fn test_gizmo_mode_default() {
        let mode: GizmoMode = Default::default();
        assert_eq!(mode, GizmoMode::Translate);
    }

    #[test]
    fn test_projection_type_default() {
        let proj: ProjectionType = Default::default();
        assert_eq!(proj, ProjectionType::Perspective);
    }

    #[test]
    fn test_light_type_default() {
        let light: LightType = Default::default();
        assert_eq!(light, LightType::Directional);
    }

    #[test]
    fn test_collision_shape_type_default() {
        let shape: CollisionShapeType = Default::default();
        assert_eq!(shape, CollisionShapeType::Box);
    }

    #[test]
    fn test_pbr_workflow_default() {
        let workflow: PbrWorkflow = Default::default();
        assert_eq!(workflow, PbrWorkflow::MetallicRoughness);
    }

    // -------------------------------------------------------------------------
    // Snap Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transform_snap_disabled() {
        let view = TransformView::new();
        assert!(!view.snap_enabled);
        let snapped = view.apply_snap(1.234, 1.0);
        assert!((snapped - 1.234).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transform_snap_enabled() {
        let mut view = TransformView::new();
        view.snap_enabled = true;
        view.snap_position = 0.5;
        let snapped = view.apply_snap(1.234, 0.5);
        assert!((snapped - 1.0).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // View Default Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transform_view_default() {
        let view: TransformView = Default::default();
        assert_eq!(view.space(), CoordinateSpace::Local);
    }

    #[test]
    fn test_material_view_default() {
        let view: MaterialView = Default::default();
        assert_eq!(view.workflow(), PbrWorkflow::MetallicRoughness);
    }

    #[test]
    fn test_mesh_view_default() {
        let view: MeshView = Default::default();
        assert!(!view.show_bounds());
    }

    #[test]
    fn test_camera_view_default() {
        let view: CameraView = Default::default();
        assert!(!view.show_frustum());
    }

    #[test]
    fn test_light_view_default() {
        let view: LightView = Default::default();
        assert!(view.show_gizmo());
    }

    #[test]
    fn test_physics_view_default() {
        let view: PhysicsView = Default::default();
        assert!(view.show_collider());
    }
}
