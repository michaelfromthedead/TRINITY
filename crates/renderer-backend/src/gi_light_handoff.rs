//! GI Light List Handoff for probe/DDGI systems.
//!
//! This module builds per-frame lists of GI-contributing lights for the
//! GI/Reflections gapset (S6). It collects lights from the scene, filters
//! by GI contributor flags, sorts by importance, and uploads to GPU buffers.
//!
//! Usage:
//! ```ignore
//! let mut handoff = GILightHandoff::new(128);
//! handoff.begin_frame();
//!
//! // Add GI-contributing lights
//! handoff.add_directional(0, &sun_light, GIImportance::Critical);
//! handoff.add_point(1, &fill_light, GIImportance::High);
//!
//! handoff.finalize();
//! handoff.upload(&device, &queue);
//!
//! // Bind buffer to GI compute pass
//! let buffer = handoff.buffer();
//! ```

use crate::light_types::{
    DirectionalLightGPU, PointLightGPU, RectAreaLightGPU, SpotLightGPU,
};
use wgpu::util::DeviceExt;

/// Minimum buffer size in bytes for empty light buffers.
/// Required because wgpu does not allow zero-sized buffers.
const MIN_BUFFER_SIZE: u64 = 16;

/// Default intensity threshold for determining GI importance.
const HIGH_INTENSITY_THRESHOLD: f32 = 10000.0;
const MEDIUM_INTENSITY_THRESHOLD: f32 = 1000.0;

/// Importance level for GI contribution (affects probe update priority).
///
/// Higher importance lights are processed first and contribute more
/// to probe updates. Critical lights (sun, key lights) always contribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u32)]
pub enum GIImportance {
    /// Ambient/accent lights - lowest priority
    Low = 0,
    /// Secondary lights - medium priority
    #[default]
    Medium = 1,
    /// Major fill lights - high priority
    High = 2,
    /// Sun, key lights - always contribute, highest priority
    Critical = 3,
}

impl GIImportance {
    /// Convert from u32 value (for GPU readback).
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Low),
            1 => Some(Self::Medium),
            2 => Some(Self::High),
            3 => Some(Self::Critical),
            _ => None,
        }
    }
}

/// Light type enumeration for GI handoff.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GILightType {
    Directional = 0,
    Point = 1,
    Spot = 2,
    Area = 3,
}

impl GILightType {
    /// Convert from u32 value.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Directional),
            1 => Some(Self::Point),
            2 => Some(Self::Spot),
            3 => Some(Self::Area),
            _ => None,
        }
    }
}

/// GI-contributing light info for handoff to probe systems.
///
/// Size: 64 bytes (16-byte aligned for GPU)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GILightInfo {
    /// Unique identifier for the light
    pub light_id: u32,
    /// Light type: 0=directional, 1=point, 2=spot, 3=area
    pub light_type: u32,
    /// GI importance level (0-3)
    pub importance: u32,
    /// Padding for alignment
    pub _padding: u32,
    /// World position of the light (directional uses direction here)
    pub position: [f32; 3],
    /// Influence radius (0 for directional)
    pub radius: f32,
    /// Light color RGB
    pub color: [f32; 3],
    /// Light intensity
    pub intensity: f32,
    /// Light direction (normalized)
    pub direction: [f32; 3],
    /// Cone angle for spot lights (0 for others)
    pub cone_angle: f32,
}

// Static size assertion: 64 bytes
const _: () = assert!(std::mem::size_of::<GILightInfo>() == 64);
const _: () = assert!(std::mem::align_of::<GILightInfo>() == 4);
const _: () = assert!(std::mem::size_of::<GILightInfo>() % 16 == 0);

impl Default for GILightInfo {
    fn default() -> Self {
        Self {
            light_id: 0,
            light_type: 0,
            importance: GIImportance::Medium as u32,
            _padding: 0,
            position: [0.0, 0.0, 0.0],
            radius: 0.0,
            color: [1.0, 1.0, 1.0],
            intensity: 0.0,
            direction: [0.0, -1.0, 0.0],
            cone_angle: 0.0,
        }
    }
}

// SAFETY: GILightInfo is #[repr(C)] with only f32/u32 fields
unsafe impl bytemuck::Pod for GILightInfo {}
unsafe impl bytemuck::Zeroable for GILightInfo {}

/// Collects GI-contributing lights for probe/DDGI systems.
///
/// This struct manages a per-frame list of lights that contribute to
/// global illumination. Lights are sorted by importance so critical
/// lights (sun, key lights) are processed first.
pub struct GILightHandoff {
    /// List of GI-contributing lights
    lights: Vec<GILightInfo>,
    /// Maximum number of lights to track
    max_lights: usize,
    /// GPU buffer for light data
    buffer: Option<wgpu::Buffer>,
    /// Whether the list has been finalized this frame
    finalized: bool,
}

impl GILightHandoff {
    /// Create a new GI light handoff with the specified maximum light count.
    ///
    /// # Arguments
    /// * `max_lights` - Maximum number of GI-contributing lights to track
    pub fn new(max_lights: usize) -> Self {
        Self {
            lights: Vec::with_capacity(max_lights),
            max_lights,
            buffer: None,
            finalized: false,
        }
    }

    /// Clear the light list for a new frame.
    ///
    /// Must be called at the start of each frame before adding lights.
    pub fn begin_frame(&mut self) {
        self.lights.clear();
        self.finalized = false;
    }

    /// Add a directional light to the GI list.
    ///
    /// Directional lights always contribute to GI (they illuminate the
    /// entire scene). They are typically marked as Critical importance.
    ///
    /// # Arguments
    /// * `light_id` - Unique identifier for the light
    /// * `light` - The directional light data
    /// * `importance` - GI importance level
    pub fn add_directional(
        &mut self,
        light_id: u32,
        light: &DirectionalLightGPU,
        importance: GIImportance,
    ) {
        if self.lights.len() >= self.max_lights {
            return;
        }

        self.lights.push(GILightInfo {
            light_id,
            light_type: GILightType::Directional as u32,
            importance: importance as u32,
            _padding: 0,
            position: light.direction, // Store direction in position for directional
            radius: 0.0, // Infinite radius
            color: light.color,
            intensity: light.intensity,
            direction: light.direction,
            cone_angle: 0.0,
        });
    }

    /// Add a point light to the GI list.
    ///
    /// # Arguments
    /// * `light_id` - Unique identifier for the light
    /// * `light` - The point light data
    /// * `importance` - GI importance level
    pub fn add_point(
        &mut self,
        light_id: u32,
        light: &PointLightGPU,
        importance: GIImportance,
    ) {
        if self.lights.len() >= self.max_lights {
            return;
        }

        self.lights.push(GILightInfo {
            light_id,
            light_type: GILightType::Point as u32,
            importance: importance as u32,
            _padding: 0,
            position: light.position,
            radius: light.radius,
            color: light.color,
            intensity: light.intensity,
            direction: [0.0, 0.0, 0.0], // Omni-directional
            cone_angle: 0.0,
        });
    }

    /// Add a spot light to the GI list.
    ///
    /// # Arguments
    /// * `light_id` - Unique identifier for the light
    /// * `light` - The spot light data
    /// * `importance` - GI importance level
    pub fn add_spot(
        &mut self,
        light_id: u32,
        light: &SpotLightGPU,
        importance: GIImportance,
    ) {
        if self.lights.len() >= self.max_lights {
            return;
        }

        self.lights.push(GILightInfo {
            light_id,
            light_type: GILightType::Spot as u32,
            importance: importance as u32,
            _padding: 0,
            position: light.position,
            radius: light.radius,
            color: light.color,
            intensity: light.intensity,
            direction: light.direction,
            cone_angle: light.outer_angle,
        });
    }

    /// Add an area light to the GI list.
    ///
    /// # Arguments
    /// * `light_id` - Unique identifier for the light
    /// * `light` - The rectangular area light data
    /// * `importance` - GI importance level
    pub fn add_area(
        &mut self,
        light_id: u32,
        light: &RectAreaLightGPU,
        importance: GIImportance,
    ) {
        if self.lights.len() >= self.max_lights {
            return;
        }

        // Use diagonal as effective radius for area lights
        let effective_radius = (light.width * light.width + light.height * light.height).sqrt();

        self.lights.push(GILightInfo {
            light_id,
            light_type: GILightType::Area as u32,
            importance: importance as u32,
            _padding: 0,
            position: light.position,
            radius: effective_radius,
            color: light.color,
            intensity: light.intensity,
            direction: light.direction,
            cone_angle: 0.0,
        });
    }

    /// Sort lights by importance (Critical first) and finalize the list.
    ///
    /// After calling this, no more lights can be added until `begin_frame`
    /// is called again.
    pub fn finalize(&mut self) {
        // Sort by importance descending (Critical=3 first, Low=0 last)
        self.lights.sort_by(|a, b| b.importance.cmp(&a.importance));
        self.finalized = true;
    }

    /// Upload the light list to the GPU buffer.
    ///
    /// Creates a new buffer if necessary, or updates the existing one.
    ///
    /// # Arguments
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue for buffer writes
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.lights.is_empty() {
            // Create minimum-sized placeholder buffer if needed
            if self.buffer.is_none() {
                self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GI Light List (empty)"),
                    size: MIN_BUFFER_SIZE,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            return;
        }

        let data = bytemuck::cast_slice(&self.lights);
        let required_size = data.len() as u64;

        // Check if we need to recreate the buffer
        let needs_new_buffer = match &self.buffer {
            Some(buf) => buf.size() < required_size,
            None => true,
        };

        if needs_new_buffer {
            self.buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GI Light List"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        } else if let Some(ref buffer) = self.buffer {
            queue.write_buffer(buffer, 0, data);
        }
    }

    /// Get the GPU buffer for GI system binding.
    ///
    /// Returns `None` if `upload` has not been called.
    pub fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Get the number of GI-contributing lights.
    pub fn count(&self) -> usize {
        self.lights.len()
    }

    /// Get the maximum number of lights this handoff can track.
    pub fn max_lights(&self) -> usize {
        self.max_lights
    }

    /// Iterate over all GI lights.
    pub fn lights(&self) -> &[GILightInfo] {
        &self.lights
    }

    /// Get lights filtered by minimum importance level.
    ///
    /// # Arguments
    /// * `min_importance` - Minimum importance level to include
    ///
    /// # Returns
    /// A vector of references to lights meeting the importance threshold.
    pub fn lights_by_importance(&self, min_importance: GIImportance) -> Vec<&GILightInfo> {
        self.lights
            .iter()
            .filter(|l| l.importance >= min_importance as u32)
            .collect()
    }

    /// Check if the light list has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}

/// Builder for collecting GI lights from a scene.
///
/// This provides a higher-level API for collecting lights from scene
/// data, automatically filtering by GI contributor flags.
pub struct GILightCollector<'a> {
    handoff: &'a mut GILightHandoff,
}

impl<'a> GILightCollector<'a> {
    /// Create a new collector targeting the given handoff.
    ///
    /// # Arguments
    /// * `handoff` - The GILightHandoff to populate
    pub fn new(handoff: &'a mut GILightHandoff) -> Self {
        Self { handoff }
    }

    /// Collect GI-contributing lights from scene light arrays.
    ///
    /// This filters lights by their `is_gi_contributor` flag and
    /// automatically computes importance based on light properties.
    ///
    /// # Arguments
    /// * `directional_lights` - Array of (id, light) tuples for directional lights
    /// * `point_lights` - Array of (id, light, is_gi_contributor) tuples
    /// * `spot_lights` - Array of (id, light, is_gi_contributor) tuples
    /// * `area_lights` - Array of (id, light, is_gi_contributor) tuples
    pub fn collect_from_scene(
        &mut self,
        directional_lights: &[(u32, DirectionalLightGPU)],
        point_lights: &[(u32, PointLightGPU, bool)],
        spot_lights: &[(u32, SpotLightGPU, bool)],
        area_lights: &[(u32, RectAreaLightGPU, bool)],
    ) {
        // Directional lights always contribute to GI
        for (id, light) in directional_lights {
            let importance = self.compute_importance(light.intensity, true);
            self.handoff.add_directional(*id, light, importance);
        }

        // Point lights filtered by GI contributor flag
        for (id, light, is_gi_contributor) in point_lights {
            if *is_gi_contributor {
                let importance = self.compute_importance(light.intensity, false);
                self.handoff.add_point(*id, light, importance);
            }
        }

        // Spot lights filtered by GI contributor flag
        for (id, light, is_gi_contributor) in spot_lights {
            if *is_gi_contributor {
                let importance = self.compute_importance(light.intensity, false);
                self.handoff.add_spot(*id, light, importance);
            }
        }

        // Area lights filtered by GI contributor flag
        for (id, light, is_gi_contributor) in area_lights {
            if *is_gi_contributor {
                let importance = self.compute_importance(light.intensity, false);
                self.handoff.add_area(*id, light, importance);
            }
        }
    }

    /// Determine importance level from light intensity and type.
    ///
    /// Directional lights are typically Critical (sun/moon).
    /// Other lights are ranked by intensity.
    fn compute_importance(&self, intensity: f32, is_directional: bool) -> GIImportance {
        if is_directional {
            // Directional lights (sun/moon) are typically critical
            GIImportance::Critical
        } else if intensity >= HIGH_INTENSITY_THRESHOLD {
            GIImportance::High
        } else if intensity >= MEDIUM_INTENSITY_THRESHOLD {
            GIImportance::Medium
        } else {
            GIImportance::Low
        }
    }

    /// Get mutable access to the underlying handoff.
    pub fn handoff_mut(&mut self) -> &mut GILightHandoff {
        self.handoff
    }

    /// Get immutable access to the underlying handoff.
    pub fn handoff(&self) -> &GILightHandoff {
        self.handoff
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gi_light_info_size() {
        // GILightInfo must be exactly 64 bytes for GPU alignment
        assert_eq!(std::mem::size_of::<GILightInfo>(), 64);
        assert_eq!(std::mem::size_of::<GILightInfo>() % 16, 0);
    }

    #[test]
    fn test_gi_light_info_alignment() {
        assert_eq!(std::mem::align_of::<GILightInfo>(), 4);
    }

    #[test]
    fn test_begin_frame_clears_list() {
        let mut handoff = GILightHandoff::new(128);

        // Add a light
        let light = DirectionalLightGPU::new_sun([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 100000.0);
        handoff.add_directional(0, &light, GIImportance::Critical);
        assert_eq!(handoff.count(), 1);

        // Begin frame should clear
        handoff.begin_frame();
        assert_eq!(handoff.count(), 0);
        assert!(!handoff.is_finalized());
    }

    #[test]
    fn test_add_directional() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let light = DirectionalLightGPU::new_sun([0.0, -1.0, 0.0], [1.0, 0.95, 0.9], 100000.0);
        handoff.add_directional(42, &light, GIImportance::Critical);

        assert_eq!(handoff.count(), 1);
        let info = &handoff.lights()[0];
        assert_eq!(info.light_id, 42);
        assert_eq!(info.light_type, GILightType::Directional as u32);
        assert_eq!(info.importance, GIImportance::Critical as u32);
        assert_eq!(info.color, [1.0, 0.95, 0.9]);
        assert_eq!(info.intensity, 100000.0);
    }

    #[test]
    fn test_add_point() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let light = PointLightGPU::new([10.0, 5.0, -3.0], [1.0, 0.8, 0.6], 800.0, 15.0);
        handoff.add_point(7, &light, GIImportance::High);

        assert_eq!(handoff.count(), 1);
        let info = &handoff.lights()[0];
        assert_eq!(info.light_id, 7);
        assert_eq!(info.light_type, GILightType::Point as u32);
        assert_eq!(info.importance, GIImportance::High as u32);
        assert_eq!(info.position, [10.0, 5.0, -3.0]);
        assert_eq!(info.radius, 15.0);
    }

    #[test]
    fn test_add_spot() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let light = SpotLightGPU::new(
            [0.0, 5.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            1000.0,
            0.4363,
            0.7854,
            20.0,
        );
        handoff.add_spot(3, &light, GIImportance::Medium);

        assert_eq!(handoff.count(), 1);
        let info = &handoff.lights()[0];
        assert_eq!(info.light_id, 3);
        assert_eq!(info.light_type, GILightType::Spot as u32);
        assert_eq!(info.direction, [0.0, -1.0, 0.0]);
        assert_eq!(info.cone_angle, 0.7854);
    }

    #[test]
    fn test_add_area() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let light = RectAreaLightGPU::new(
            [0.0, 3.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            2.0,
            1.0,
            [1.0, 1.0, 1.0],
            500.0,
        );
        handoff.add_area(5, &light, GIImportance::Low);

        assert_eq!(handoff.count(), 1);
        let info = &handoff.lights()[0];
        assert_eq!(info.light_id, 5);
        assert_eq!(info.light_type, GILightType::Area as u32);
        assert_eq!(info.importance, GIImportance::Low as u32);
        // Effective radius = sqrt(2^2 + 1^2) = sqrt(5) ~ 2.236
        assert!((info.radius - 5.0_f32.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_finalize_sorts_by_importance() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        // Add lights in reverse importance order
        let point = PointLightGPU::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 100.0, 5.0);
        handoff.add_point(1, &point, GIImportance::Low);
        handoff.add_point(2, &point, GIImportance::High);
        handoff.add_point(3, &point, GIImportance::Medium);

        let sun = DirectionalLightGPU::new_sun([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 100000.0);
        handoff.add_directional(4, &sun, GIImportance::Critical);

        handoff.finalize();

        // Should be sorted: Critical, High, Medium, Low
        assert!(handoff.is_finalized());
        assert_eq!(handoff.lights()[0].importance, GIImportance::Critical as u32);
        assert_eq!(handoff.lights()[1].importance, GIImportance::High as u32);
        assert_eq!(handoff.lights()[2].importance, GIImportance::Medium as u32);
        assert_eq!(handoff.lights()[3].importance, GIImportance::Low as u32);
    }

    #[test]
    fn test_count_matches_added_lights() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let sun = DirectionalLightGPU::new_sun([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 100000.0);
        let point = PointLightGPU::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 100.0, 5.0);
        let spot = SpotLightGPU::new(
            [0.0, 5.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            1000.0,
            0.4,
            0.8,
            20.0,
        );

        handoff.add_directional(0, &sun, GIImportance::Critical);
        assert_eq!(handoff.count(), 1);

        handoff.add_point(1, &point, GIImportance::Medium);
        assert_eq!(handoff.count(), 2);

        handoff.add_spot(2, &spot, GIImportance::High);
        assert_eq!(handoff.count(), 3);
    }

    #[test]
    fn test_max_lights_limit_enforced() {
        let mut handoff = GILightHandoff::new(3);
        handoff.begin_frame();

        let point = PointLightGPU::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 100.0, 5.0);

        // Add 5 lights, but max is 3
        for i in 0..5 {
            handoff.add_point(i, &point, GIImportance::Medium);
        }

        // Should only have 3 lights
        assert_eq!(handoff.count(), 3);
        assert_eq!(handoff.max_lights(), 3);
    }

    #[test]
    fn test_lights_by_importance_filters_correctly() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let point = PointLightGPU::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 100.0, 5.0);
        handoff.add_point(1, &point, GIImportance::Low);
        handoff.add_point(2, &point, GIImportance::Medium);
        handoff.add_point(3, &point, GIImportance::High);
        handoff.add_point(4, &point, GIImportance::Critical);

        // Filter by High or better
        let high_plus = handoff.lights_by_importance(GIImportance::High);
        assert_eq!(high_plus.len(), 2);
        assert!(high_plus.iter().all(|l| l.importance >= GIImportance::High as u32));

        // Filter by Critical only
        let critical = handoff.lights_by_importance(GIImportance::Critical);
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].light_id, 4);

        // Filter by Low or better (all)
        let all = handoff.lights_by_importance(GIImportance::Low);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_collector_processes_scene_correctly() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let sun = DirectionalLightGPU::new_sun([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 100000.0);
        let point1 = PointLightGPU::new([5.0, 2.0, 0.0], [1.0, 0.8, 0.6], 5000.0, 10.0);
        let point2 = PointLightGPU::new([0.0, 2.0, 5.0], [0.6, 0.8, 1.0], 200.0, 8.0);
        let spot = SpotLightGPU::new(
            [0.0, 5.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 1.0],
            15000.0,
            0.4,
            0.8,
            20.0,
        );

        let directional_lights = vec![(0, sun)];
        let point_lights = vec![
            (1, point1, true),  // GI contributor
            (2, point2, false), // NOT a GI contributor
        ];
        let spot_lights = vec![(3, spot, true)];
        let area_lights: Vec<(u32, RectAreaLightGPU, bool)> = vec![];

        let mut collector = GILightCollector::new(&mut handoff);
        collector.collect_from_scene(
            &directional_lights,
            &point_lights,
            &spot_lights,
            &area_lights,
        );

        // Should have 3 lights (sun, point1, spot) - point2 was excluded
        assert_eq!(handoff.count(), 3);

        // Verify light IDs present
        let ids: Vec<u32> = handoff.lights().iter().map(|l| l.light_id).collect();
        assert!(ids.contains(&0)); // sun
        assert!(ids.contains(&1)); // point1 (GI contributor)
        assert!(!ids.contains(&2)); // point2 should be excluded
        assert!(ids.contains(&3)); // spot
    }

    #[test]
    fn test_non_gi_contributor_lights_excluded() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let point = PointLightGPU::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 100.0, 5.0);

        let directional_lights: Vec<(u32, DirectionalLightGPU)> = vec![];
        let point_lights = vec![
            (1, point, false), // NOT a GI contributor
            (2, point, false), // NOT a GI contributor
            (3, point, true),  // IS a GI contributor
        ];
        let spot_lights: Vec<(u32, SpotLightGPU, bool)> = vec![];
        let area_lights: Vec<(u32, RectAreaLightGPU, bool)> = vec![];

        let mut collector = GILightCollector::new(&mut handoff);
        collector.collect_from_scene(
            &directional_lights,
            &point_lights,
            &spot_lights,
            &area_lights,
        );

        // Only light ID 3 should be present
        assert_eq!(handoff.count(), 1);
        assert_eq!(handoff.lights()[0].light_id, 3);
    }

    #[test]
    fn test_compute_importance_thresholds() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let collector = GILightCollector::new(&mut handoff);

        // Directional lights are always Critical
        assert_eq!(collector.compute_importance(1000.0, true), GIImportance::Critical);
        assert_eq!(collector.compute_importance(100.0, true), GIImportance::Critical);

        // High intensity (>= 10000)
        assert_eq!(collector.compute_importance(10000.0, false), GIImportance::High);
        assert_eq!(collector.compute_importance(50000.0, false), GIImportance::High);

        // Medium intensity (>= 1000, < 10000)
        assert_eq!(collector.compute_importance(1000.0, false), GIImportance::Medium);
        assert_eq!(collector.compute_importance(5000.0, false), GIImportance::Medium);

        // Low intensity (< 1000)
        assert_eq!(collector.compute_importance(999.0, false), GIImportance::Low);
        assert_eq!(collector.compute_importance(100.0, false), GIImportance::Low);
    }

    #[test]
    fn test_gi_importance_ordering() {
        // Verify importance levels are ordered correctly
        assert!(GIImportance::Critical > GIImportance::High);
        assert!(GIImportance::High > GIImportance::Medium);
        assert!(GIImportance::Medium > GIImportance::Low);
    }

    #[test]
    fn test_gi_importance_from_u32() {
        assert_eq!(GIImportance::from_u32(0), Some(GIImportance::Low));
        assert_eq!(GIImportance::from_u32(1), Some(GIImportance::Medium));
        assert_eq!(GIImportance::from_u32(2), Some(GIImportance::High));
        assert_eq!(GIImportance::from_u32(3), Some(GIImportance::Critical));
        assert_eq!(GIImportance::from_u32(4), None);
        assert_eq!(GIImportance::from_u32(255), None);
    }

    #[test]
    fn test_gi_light_type_from_u32() {
        assert_eq!(GILightType::from_u32(0), Some(GILightType::Directional));
        assert_eq!(GILightType::from_u32(1), Some(GILightType::Point));
        assert_eq!(GILightType::from_u32(2), Some(GILightType::Spot));
        assert_eq!(GILightType::from_u32(3), Some(GILightType::Area));
        assert_eq!(GILightType::from_u32(4), None);
    }

    #[test]
    fn test_bytemuck_traits() {
        // Verify bytemuck can cast GILightInfo to bytes
        let info = GILightInfo::default();
        let bytes: &[u8] = bytemuck::bytes_of(&info);
        assert_eq!(bytes.len(), 64);

        // Verify we can create zeroed instances
        let zeroed: GILightInfo = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.light_id, 0);
        assert_eq!(zeroed.position, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_empty_handoff() {
        let handoff = GILightHandoff::new(128);
        assert_eq!(handoff.count(), 0);
        assert!(handoff.lights().is_empty());
        assert!(!handoff.is_finalized());
        assert!(handoff.buffer().is_none());
    }

    #[test]
    fn test_gi_light_info_default() {
        let info = GILightInfo::default();
        assert_eq!(info.light_id, 0);
        assert_eq!(info.light_type, 0);
        assert_eq!(info.importance, GIImportance::Medium as u32);
        assert_eq!(info.color, [1.0, 1.0, 1.0]);
        assert_eq!(info.intensity, 0.0);
        assert_eq!(info.direction, [0.0, -1.0, 0.0]);
    }

    #[test]
    fn test_collector_access_to_handoff() {
        let mut handoff = GILightHandoff::new(128);
        handoff.begin_frame();

        let sun = DirectionalLightGPU::new_sun([0.0, -1.0, 0.0], [1.0, 1.0, 1.0], 100000.0);

        {
            let mut collector = GILightCollector::new(&mut handoff);

            // Access handoff through collector
            assert_eq!(collector.handoff().count(), 0);

            // Add light through handoff_mut
            collector.handoff_mut().add_directional(0, &sun, GIImportance::Critical);
            assert_eq!(collector.handoff().count(), 1);
        }

        // Verify changes persisted
        assert_eq!(handoff.count(), 1);
    }
}
