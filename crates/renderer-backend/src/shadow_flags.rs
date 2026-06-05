//! Cast/Receive shadow flags for per-object shadow control (T-LIT-9.5).
//!
//! This module provides fine-grained control over shadow behavior for each
//! renderable object. Objects can independently cast shadows, receive shadows,
//! contribute to contact shadows, and participate in global illumination.
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::shadow_flags::{ShadowFlags, ShadowFlagBuffer, ShadowFlagsGpu};
//!
//! // Create flags for different object types
//! let character_flags = ShadowFlags::DYNAMIC;  // cast + receive + contact
//! let skybox_flags = ShadowFlags::RECEIVE_ONLY;  // receive only
//! let terrain_flags = ShadowFlags::STATIC_WORLD;  // all flags
//!
//! // Create GPU buffer for all instances
//! let mut flag_buffer = ShadowFlagBuffer::new(&device, 1000);
//! flag_buffer.set_flags(0, character_flags);
//! flag_buffer.set_bias(0, 0.001);  // Per-object shadow bias
//! flag_buffer.upload(&queue);
//! ```
//!
//! # GPU Struct Layout
//!
//! The `ShadowFlagsGpu` struct is 16 bytes, matching std140/std430 alignment:
//!
//! | Offset | Field             | Type   | Size    |
//! |--------|-------------------|--------|---------|
//! | 0      | flags             | u32    | 4 bytes |
//! | 4      | shadow_bias       | f32    | 4 bytes |
//! | 8      | shadow_fade_start | f32    | 4 bytes |
//! | 12     | shadow_fade_end   | f32    | 4 bytes |
//!
//! # WGSL Integration
//!
//! ```wgsl
//! struct ShadowFlagsGpu {
//!     flags: u32,
//!     shadow_bias: f32,
//!     shadow_fade_start: f32,
//!     shadow_fade_end: f32,
//! }
//!
//! const CAST_SHADOW: u32 = 0x0001u;
//! const RECEIVE_SHADOW: u32 = 0x0002u;
//! const CONTACT_SHADOW: u32 = 0x0004u;
//! const GI_CONTRIBUTOR: u32 = 0x0008u;
//!
//! fn apply_shadow(
//!     instance_id: u32,
//!     shadow_factor: f32,
//!     camera_distance: f32,
//!     flags_buffer: ptr<storage, array<ShadowFlagsGpu>>
//! ) -> f32 {
//!     let data = (*flags_buffer)[instance_id];
//!     if (data.flags & RECEIVE_SHADOW) == 0u {
//!         return 1.0; // Object doesn't receive shadows
//!     }
//!     // Apply distance fade
//!     let fade = smoothstep(data.shadow_fade_end, data.shadow_fade_start, camera_distance);
//!     return mix(1.0, shadow_factor, fade);
//! }
//! ```

use std::mem::size_of;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// ShadowFlags Bitflags
// ---------------------------------------------------------------------------

bitflags::bitflags! {
    /// Shadow behavior flags for renderable objects.
    ///
    /// Each flag controls a specific aspect of shadow rendering:
    /// - `CAST_SHADOW`: Object casts shadows onto other objects
    /// - `RECEIVE_SHADOW`: Object receives shadows from other objects
    /// - `CONTACT_SHADOW`: Object is included in contact shadow ray march
    /// - `GI_CONTRIBUTOR`: Object contributes to global illumination (DDGI/probes)
    ///
    /// Several preset combinations are provided for common use cases.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct ShadowFlags: u32 {
        /// Object casts shadows onto others.
        const CAST_SHADOW = 0b0001;

        /// Object receives shadows from others.
        const RECEIVE_SHADOW = 0b0010;

        /// Object is included in contact shadow ray march.
        const CONTACT_SHADOW = 0b0100;

        /// Object contributes to GI (for DDGI/probe systems).
        const GI_CONTRIBUTOR = 0b1000;

        // --- Preset Combinations ---

        /// Default for most geometry (cast + receive).
        const DEFAULT = Self::CAST_SHADOW.bits() | Self::RECEIVE_SHADOW.bits();

        /// Background/skybox (receive only, no casting).
        const RECEIVE_ONLY = Self::RECEIVE_SHADOW.bits();

        /// Characters/dynamic objects (cast + receive + contact).
        const DYNAMIC = Self::CAST_SHADOW.bits() | Self::RECEIVE_SHADOW.bits() | Self::CONTACT_SHADOW.bits();

        /// Static world geometry (all flags enabled).
        const STATIC_WORLD = Self::CAST_SHADOW.bits() | Self::RECEIVE_SHADOW.bits() | Self::CONTACT_SHADOW.bits() | Self::GI_CONTRIBUTOR.bits();
    }
}

impl ShadowFlags {
    /// Returns `true` if the object casts shadows.
    #[inline]
    pub fn casts_shadow(self) -> bool {
        self.contains(Self::CAST_SHADOW)
    }

    /// Returns `true` if the object receives shadows.
    #[inline]
    pub fn receives_shadow(self) -> bool {
        self.contains(Self::RECEIVE_SHADOW)
    }

    /// Returns `true` if the object participates in contact shadow ray marching.
    #[inline]
    pub fn contacts_shadow(self) -> bool {
        self.contains(Self::CONTACT_SHADOW)
    }

    /// Returns `true` if the object contributes to global illumination.
    #[inline]
    pub fn contributes_gi(self) -> bool {
        self.contains(Self::GI_CONTRIBUTOR)
    }

    /// Create flags from raw u32 bits.
    ///
    /// This is useful for reading flags from GPU readback or serialized data.
    #[inline]
    pub fn from_raw(bits: u32) -> Self {
        Self::from_bits_truncate(bits)
    }
}

// ---------------------------------------------------------------------------
// GPU-Compatible Struct
// ---------------------------------------------------------------------------

/// GPU-compatible shadow flag data (per-instance).
///
/// This struct is designed for direct upload to a storage buffer.
/// Size: 16 bytes (4x f32), aligned for efficient GPU access.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowFlagsGpu {
    /// Shadow behavior flags (see [`ShadowFlags`]).
    pub flags: u32,

    /// Per-object shadow bias override.
    /// A value of 0.0 indicates use of the global bias.
    pub shadow_bias: f32,

    /// Distance from camera where shadow starts fading (world units).
    pub shadow_fade_start: f32,

    /// Distance from camera where shadow is fully faded (world units).
    pub shadow_fade_end: f32,
}

// Compile-time size assertion
const _: () = assert!(size_of::<ShadowFlagsGpu>() == 16);

impl Default for ShadowFlagsGpu {
    fn default() -> Self {
        Self {
            flags: ShadowFlags::DEFAULT.bits(),
            shadow_bias: 0.0, // Use global bias
            shadow_fade_start: 50.0,
            shadow_fade_end: 100.0,
        }
    }
}

impl ShadowFlagsGpu {
    /// Create GPU flags from [`ShadowFlags`] with default fade distances.
    pub fn from_flags(flags: ShadowFlags) -> Self {
        Self {
            flags: flags.bits(),
            ..Default::default()
        }
    }

    /// Create GPU flags with custom fade distances.
    pub fn with_fade(flags: ShadowFlags, fade_start: f32, fade_end: f32) -> Self {
        Self {
            flags: flags.bits(),
            shadow_bias: 0.0,
            shadow_fade_start: fade_start,
            shadow_fade_end: fade_end,
        }
    }

    /// Create GPU flags with custom bias and fade distances.
    pub fn new(flags: ShadowFlags, bias: f32, fade_start: f32, fade_end: f32) -> Self {
        Self {
            flags: flags.bits(),
            shadow_bias: bias,
            shadow_fade_start: fade_start,
            shadow_fade_end: fade_end,
        }
    }

    /// Returns the [`ShadowFlags`] from the raw bits.
    pub fn shadow_flags(&self) -> ShadowFlags {
        ShadowFlags::from_raw(self.flags)
    }
}

// ---------------------------------------------------------------------------
// Shadow Flag Buffer
// ---------------------------------------------------------------------------

/// Shadow flag buffer for all instances.
///
/// Manages a GPU storage buffer containing per-instance shadow configuration.
/// The buffer can be bound to shaders to control per-object shadow behavior.
///
/// # Example
///
/// ```ignore
/// let mut buffer = ShadowFlagBuffer::new(&device, 1000);
///
/// // Configure instance 0 as a dynamic character
/// buffer.set_flags(0, ShadowFlags::DYNAMIC);
/// buffer.set_bias(0, 0.001);
///
/// // Configure instance 1 as static world geometry
/// buffer.set_flags(1, ShadowFlags::STATIC_WORLD);
/// buffer.set_fade_distances(1, 100.0, 200.0);
///
/// // Upload to GPU
/// buffer.upload(&queue);
/// ```
pub struct ShadowFlagBuffer {
    buffer: wgpu::Buffer,
    data: Vec<ShadowFlagsGpu>,
    capacity: usize,
    dirty: bool,
}

impl ShadowFlagBuffer {
    /// Create a new shadow flag buffer with the given capacity.
    ///
    /// All instances are initialized with [`ShadowFlagsGpu::default()`].
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for buffer creation.
    /// * `capacity` - Maximum number of instances.
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let capacity = capacity.max(1); // At least 1 element
        let data = vec![ShadowFlagsGpu::default(); capacity];

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShadowFlagBuffer"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            buffer,
            data,
            capacity,
            dirty: false,
        }
    }

    /// Create an empty shadow flag buffer (for testing or deferred initialization).
    ///
    /// The buffer has a minimum size of 16 bytes.
    pub fn empty(device: &wgpu::Device) -> Self {
        Self::new(device, 1)
    }

    /// Set the shadow flags for an instance.
    ///
    /// # Panics
    ///
    /// Panics if `instance_id >= capacity`.
    pub fn set_flags(&mut self, instance_id: usize, flags: ShadowFlags) {
        assert!(
            instance_id < self.capacity,
            "instance_id {} out of bounds (capacity {})",
            instance_id,
            self.capacity
        );
        self.data[instance_id].flags = flags.bits();
        self.dirty = true;
    }

    /// Set the shadow bias for an instance.
    ///
    /// A bias of 0.0 indicates use of the global shadow bias.
    ///
    /// # Panics
    ///
    /// Panics if `instance_id >= capacity`.
    pub fn set_bias(&mut self, instance_id: usize, bias: f32) {
        assert!(
            instance_id < self.capacity,
            "instance_id {} out of bounds (capacity {})",
            instance_id,
            self.capacity
        );
        self.data[instance_id].shadow_bias = bias;
        self.dirty = true;
    }

    /// Set the shadow fade distances for an instance.
    ///
    /// # Arguments
    ///
    /// * `instance_id` - Instance index.
    /// * `start` - Distance where shadow starts fading.
    /// * `end` - Distance where shadow is fully faded.
    ///
    /// # Panics
    ///
    /// Panics if `instance_id >= capacity`.
    pub fn set_fade_distances(&mut self, instance_id: usize, start: f32, end: f32) {
        assert!(
            instance_id < self.capacity,
            "instance_id {} out of bounds (capacity {})",
            instance_id,
            self.capacity
        );
        self.data[instance_id].shadow_fade_start = start;
        self.data[instance_id].shadow_fade_end = end;
        self.dirty = true;
    }

    /// Set all properties for an instance at once.
    ///
    /// # Panics
    ///
    /// Panics if `instance_id >= capacity`.
    pub fn set(&mut self, instance_id: usize, gpu_flags: ShadowFlagsGpu) {
        assert!(
            instance_id < self.capacity,
            "instance_id {} out of bounds (capacity {})",
            instance_id,
            self.capacity
        );
        self.data[instance_id] = gpu_flags;
        self.dirty = true;
    }

    /// Get the shadow flags for an instance.
    ///
    /// # Panics
    ///
    /// Panics if `instance_id >= capacity`.
    pub fn get(&self, instance_id: usize) -> &ShadowFlagsGpu {
        assert!(
            instance_id < self.capacity,
            "instance_id {} out of bounds (capacity {})",
            instance_id,
            self.capacity
        );
        &self.data[instance_id]
    }

    /// Upload the CPU-side data to the GPU buffer.
    ///
    /// This should be called after modifying flags and before rendering.
    /// Only uploads if data has been modified since last upload.
    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if self.dirty {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.data));
            self.dirty = false;
        }
    }

    /// Force upload even if not dirty (useful after resize).
    pub fn force_upload(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.data));
        self.dirty = false;
    }

    /// Get the underlying wgpu buffer for binding.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Returns the capacity (maximum number of instances).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true if data has been modified since last upload.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Returns the buffer size in bytes.
    pub fn size_bytes(&self) -> u64 {
        self.buffer.size()
    }

    /// Reset all instances to default flags.
    pub fn reset_all(&mut self) {
        for data in &mut self.data {
            *data = ShadowFlagsGpu::default();
        }
        self.dirty = true;
    }

    /// Resize the buffer to a new capacity.
    ///
    /// If the new capacity is larger, new instances are initialized with defaults.
    /// If smaller, excess instances are dropped.
    pub fn resize(&mut self, device: &wgpu::Device, new_capacity: usize) {
        let new_capacity = new_capacity.max(1);
        if new_capacity == self.capacity {
            return;
        }

        self.data.resize(new_capacity, ShadowFlagsGpu::default());
        self.capacity = new_capacity;

        // Create new buffer
        self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShadowFlagBuffer"),
            contents: bytemuck::cast_slice(&self.data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        self.dirty = false; // Just uploaded via create_buffer_init
    }
}

// ---------------------------------------------------------------------------
// Filter Functions
// ---------------------------------------------------------------------------

/// Filter objects that cast shadows.
///
/// Returns a vector of references to objects where `get_flags(obj).casts_shadow()` is true.
///
/// # Example
///
/// ```ignore
/// let casters = filter_shadow_casters(
///     scene_objects.iter(),
///     |obj| obj.shadow_flags,
/// );
/// ```
pub fn filter_shadow_casters<'a, T>(
    objects: impl Iterator<Item = &'a T>,
    get_flags: impl Fn(&T) -> ShadowFlags,
) -> Vec<&'a T>
where
    T: 'a,
{
    objects.filter(|obj| get_flags(obj).casts_shadow()).collect()
}

/// Filter objects that receive shadows.
///
/// Returns a vector of references to objects where `get_flags(obj).receives_shadow()` is true.
pub fn filter_shadow_receivers<'a, T>(
    objects: impl Iterator<Item = &'a T>,
    get_flags: impl Fn(&T) -> ShadowFlags,
) -> Vec<&'a T>
where
    T: 'a,
{
    objects
        .filter(|obj| get_flags(obj).receives_shadow())
        .collect()
}

/// Filter objects that participate in contact shadows.
///
/// Returns a vector of references to objects where `get_flags(obj).contacts_shadow()` is true.
pub fn filter_contact_shadow_objects<'a, T>(
    objects: impl Iterator<Item = &'a T>,
    get_flags: impl Fn(&T) -> ShadowFlags,
) -> Vec<&'a T>
where
    T: 'a,
{
    objects
        .filter(|obj| get_flags(obj).contacts_shadow())
        .collect()
}

/// Filter objects that contribute to global illumination.
///
/// Returns a vector of references to objects where `get_flags(obj).contributes_gi()` is true.
pub fn filter_gi_contributors<'a, T>(
    objects: impl Iterator<Item = &'a T>,
    get_flags: impl Fn(&T) -> ShadowFlags,
) -> Vec<&'a T>
where
    T: 'a,
{
    objects
        .filter(|obj| get_flags(obj).contributes_gi())
        .collect()
}

/// Partition objects into casters and non-casters.
///
/// Returns (casters, non_casters) where casters is objects that cast shadows.
pub fn partition_shadow_casters<'a, T>(
    objects: impl Iterator<Item = &'a T>,
    get_flags: impl Fn(&T) -> ShadowFlags,
) -> (Vec<&'a T>, Vec<&'a T>)
where
    T: 'a,
{
    let (casters, non_casters): (Vec<_>, Vec<_>) =
        objects.partition(|obj| get_flags(obj).casts_shadow());
    (casters, non_casters)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Default flags are CAST | RECEIVE
    #[test]
    fn test_default_flags() {
        let flags = ShadowFlags::DEFAULT;
        assert!(flags.casts_shadow());
        assert!(flags.receives_shadow());
        assert!(!flags.contacts_shadow());
        assert!(!flags.contributes_gi());
    }

    // Test 2: Bitflag operations work correctly
    #[test]
    fn test_bitflag_operations() {
        // OR operation
        let flags = ShadowFlags::CAST_SHADOW | ShadowFlags::CONTACT_SHADOW;
        assert!(flags.casts_shadow());
        assert!(!flags.receives_shadow());
        assert!(flags.contacts_shadow());
        assert!(!flags.contributes_gi());

        // AND operation
        let combined = ShadowFlags::DYNAMIC & ShadowFlags::DEFAULT;
        assert!(combined.casts_shadow());
        assert!(combined.receives_shadow());
        assert!(!combined.contacts_shadow());

        // NOT operation
        let inverted = !ShadowFlags::CAST_SHADOW;
        assert!(!inverted.contains(ShadowFlags::CAST_SHADOW));
        assert!(inverted.contains(ShadowFlags::RECEIVE_SHADOW));

        // Contains check
        assert!(ShadowFlags::DYNAMIC.contains(ShadowFlags::CAST_SHADOW));
        assert!(ShadowFlags::DYNAMIC.contains(ShadowFlags::RECEIVE_SHADOW));
        assert!(ShadowFlags::DYNAMIC.contains(ShadowFlags::CONTACT_SHADOW));
        assert!(!ShadowFlags::DYNAMIC.contains(ShadowFlags::GI_CONTRIBUTOR));
    }

    // Test 3: GPU struct is 16 bytes
    #[test]
    fn test_gpu_struct_size() {
        assert_eq!(size_of::<ShadowFlagsGpu>(), 16);
    }

    // Test 4: Preset flags (DYNAMIC, STATIC_WORLD) correct
    #[test]
    fn test_preset_flags() {
        // RECEIVE_ONLY
        let receive_only = ShadowFlags::RECEIVE_ONLY;
        assert!(!receive_only.casts_shadow());
        assert!(receive_only.receives_shadow());
        assert!(!receive_only.contacts_shadow());
        assert!(!receive_only.contributes_gi());

        // DYNAMIC
        let dynamic = ShadowFlags::DYNAMIC;
        assert!(dynamic.casts_shadow());
        assert!(dynamic.receives_shadow());
        assert!(dynamic.contacts_shadow());
        assert!(!dynamic.contributes_gi());

        // STATIC_WORLD
        let static_world = ShadowFlags::STATIC_WORLD;
        assert!(static_world.casts_shadow());
        assert!(static_world.receives_shadow());
        assert!(static_world.contacts_shadow());
        assert!(static_world.contributes_gi());
    }

    // Test 5: ShadowFlagsGpu default values
    #[test]
    fn test_gpu_flags_default() {
        let gpu = ShadowFlagsGpu::default();
        assert_eq!(gpu.flags, ShadowFlags::DEFAULT.bits());
        assert_eq!(gpu.shadow_bias, 0.0);
        assert_eq!(gpu.shadow_fade_start, 50.0);
        assert_eq!(gpu.shadow_fade_end, 100.0);
    }

    // Test 6: ShadowFlagsGpu constructors
    #[test]
    fn test_gpu_flags_constructors() {
        let from_flags = ShadowFlagsGpu::from_flags(ShadowFlags::DYNAMIC);
        assert_eq!(from_flags.flags, ShadowFlags::DYNAMIC.bits());
        assert_eq!(from_flags.shadow_bias, 0.0);

        let with_fade = ShadowFlagsGpu::with_fade(ShadowFlags::STATIC_WORLD, 25.0, 75.0);
        assert_eq!(with_fade.flags, ShadowFlags::STATIC_WORLD.bits());
        assert_eq!(with_fade.shadow_fade_start, 25.0);
        assert_eq!(with_fade.shadow_fade_end, 75.0);

        let full = ShadowFlagsGpu::new(ShadowFlags::RECEIVE_ONLY, 0.005, 30.0, 60.0);
        assert_eq!(full.flags, ShadowFlags::RECEIVE_ONLY.bits());
        assert_eq!(full.shadow_bias, 0.005);
        assert_eq!(full.shadow_fade_start, 30.0);
        assert_eq!(full.shadow_fade_end, 60.0);
    }

    // Test 7: shadow_flags() conversion
    #[test]
    fn test_gpu_flags_to_shadow_flags() {
        let gpu = ShadowFlagsGpu::from_flags(ShadowFlags::DYNAMIC);
        let flags = gpu.shadow_flags();
        assert_eq!(flags, ShadowFlags::DYNAMIC);
    }

    // Test 8: from_raw truncates invalid bits
    #[test]
    fn test_from_raw_truncation() {
        let flags = ShadowFlags::from_raw(0xFFFF);
        // Should only keep valid bits (0-15)
        assert!(flags.casts_shadow());
        assert!(flags.receives_shadow());
        assert!(flags.contacts_shadow());
        assert!(flags.contributes_gi());
    }

    // Test 9: Filter casters excludes non-casters
    #[test]
    fn test_filter_casters() {
        struct TestObj {
            id: u32,
            flags: ShadowFlags,
        }

        let objects = vec![
            TestObj {
                id: 0,
                flags: ShadowFlags::DYNAMIC,
            },
            TestObj {
                id: 1,
                flags: ShadowFlags::RECEIVE_ONLY,
            },
            TestObj {
                id: 2,
                flags: ShadowFlags::STATIC_WORLD,
            },
            TestObj {
                id: 3,
                flags: ShadowFlags::empty(),
            },
        ];

        let casters = filter_shadow_casters(objects.iter(), |obj| obj.flags);

        assert_eq!(casters.len(), 2);
        assert_eq!(casters[0].id, 0);
        assert_eq!(casters[1].id, 2);
    }

    // Test 10: Filter receivers excludes non-receivers
    #[test]
    fn test_filter_receivers() {
        struct TestObj {
            id: u32,
            flags: ShadowFlags,
        }

        let objects = vec![
            TestObj {
                id: 0,
                flags: ShadowFlags::DYNAMIC,
            },
            TestObj {
                id: 1,
                flags: ShadowFlags::CAST_SHADOW,
            }, // Cast only
            TestObj {
                id: 2,
                flags: ShadowFlags::RECEIVE_ONLY,
            },
            TestObj {
                id: 3,
                flags: ShadowFlags::empty(),
            },
        ];

        let receivers = filter_shadow_receivers(objects.iter(), |obj| obj.flags);

        assert_eq!(receivers.len(), 2);
        assert_eq!(receivers[0].id, 0);
        assert_eq!(receivers[1].id, 2);
    }

    // Test 11: Filter contact shadow objects
    #[test]
    fn test_filter_contact_shadow() {
        struct TestObj {
            id: u32,
            flags: ShadowFlags,
        }

        let objects = vec![
            TestObj {
                id: 0,
                flags: ShadowFlags::DYNAMIC,
            }, // Has contact
            TestObj {
                id: 1,
                flags: ShadowFlags::DEFAULT,
            }, // No contact
            TestObj {
                id: 2,
                flags: ShadowFlags::STATIC_WORLD,
            }, // Has contact
        ];

        let contacts = filter_contact_shadow_objects(objects.iter(), |obj| obj.flags);

        assert_eq!(contacts.len(), 2);
        assert_eq!(contacts[0].id, 0);
        assert_eq!(contacts[1].id, 2);
    }

    // Test 12: Filter GI contributors
    #[test]
    fn test_filter_gi_contributors() {
        struct TestObj {
            id: u32,
            flags: ShadowFlags,
        }

        let objects = vec![
            TestObj {
                id: 0,
                flags: ShadowFlags::DYNAMIC,
            },
            TestObj {
                id: 1,
                flags: ShadowFlags::STATIC_WORLD,
            }, // Has GI
            TestObj {
                id: 2,
                flags: ShadowFlags::GI_CONTRIBUTOR,
            }, // Has GI
        ];

        let gi = filter_gi_contributors(objects.iter(), |obj| obj.flags);

        assert_eq!(gi.len(), 2);
        assert_eq!(gi[0].id, 1);
        assert_eq!(gi[1].id, 2);
    }

    // Test 13: Partition casters
    #[test]
    fn test_partition_casters() {
        struct TestObj {
            id: u32,
            flags: ShadowFlags,
        }

        let objects = vec![
            TestObj {
                id: 0,
                flags: ShadowFlags::CAST_SHADOW,
            },
            TestObj {
                id: 1,
                flags: ShadowFlags::RECEIVE_ONLY,
            },
            TestObj {
                id: 2,
                flags: ShadowFlags::DYNAMIC,
            },
        ];

        let (casters, non_casters) = partition_shadow_casters(objects.iter(), |obj| obj.flags);

        assert_eq!(casters.len(), 2);
        assert_eq!(non_casters.len(), 1);
        assert_eq!(non_casters[0].id, 1);
    }

    // Test 14: Empty flags
    #[test]
    fn test_empty_flags() {
        let flags = ShadowFlags::empty();
        assert!(!flags.casts_shadow());
        assert!(!flags.receives_shadow());
        assert!(!flags.contacts_shadow());
        assert!(!flags.contributes_gi());
        assert_eq!(flags.bits(), 0);
    }

    // Test 15: All flags
    #[test]
    fn test_all_flags() {
        let flags = ShadowFlags::all();
        assert!(flags.casts_shadow());
        assert!(flags.receives_shadow());
        assert!(flags.contacts_shadow());
        assert!(flags.contributes_gi());
    }

    // Test 16: Flag bit values match documentation
    #[test]
    fn test_flag_bit_values() {
        assert_eq!(ShadowFlags::CAST_SHADOW.bits(), 0b0001);
        assert_eq!(ShadowFlags::RECEIVE_SHADOW.bits(), 0b0010);
        assert_eq!(ShadowFlags::CONTACT_SHADOW.bits(), 0b0100);
        assert_eq!(ShadowFlags::GI_CONTRIBUTOR.bits(), 0b1000);
    }
}
