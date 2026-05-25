use bytemuck::{Pod, Zeroable};
use core::fmt;

/// Maximum number of live entities (`2^24 - 1`).
pub const MAX_ENTITIES: u32 = (1 << 24) - 1;

/// 24-bit index mask (bits 0-23).
const INDEX_MASK: u32 = MAX_ENTITIES;

/// A generational index used to identify entities.
///
/// Packed into a single `u32`:
///
/// | Offset | Size | Field        |
/// |--------|------|--------------|
/// | 0–23   | 24   | Entity index |
/// | 24–31  |  8   | Generation   |
///
/// The generation is incremented when an entity slot is reused after despawning,
/// which allows the system to detect and reject stale handles.
///
/// # Sentinel
///
/// [`EntityId::NULL`] (`u32::MAX`) represents an invalid / null entity.
/// It is the default value and can be tested with [`EntityId::is_null`].
///
/// # `bytemuck` compatibility
///
/// `EntityId` derives [`Pod`] and [`Zeroable`], so it can be safely
/// transmuted from/to raw `u32` and used in GPU buffers.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct EntityId(u32);

impl EntityId {
    /// The null / invalid entity sentinel (`u32::MAX`).
    pub const NULL: EntityId = EntityId(u32::MAX);

    /// Creates a new `EntityId` from an index and generation.
    ///
    /// Only the lower 24 bits of `index` and the lower 8 bits of `generation`
    /// are used; any higher bits are discarded.
    #[inline]
    pub const fn new(index: u32, generation: u8) -> Self {
        EntityId((index & INDEX_MASK) | ((generation as u32) << 24))
    }

    /// Returns the 24-bit index portion of this entity ID.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0 & INDEX_MASK
    }

    /// Returns the 8-bit generation portion of this entity ID.
    #[inline]
    pub const fn generation(self) -> u8 {
        (self.0 >> 24) as u8
    }

    /// Returns `true` if this entity ID is the null sentinel (`u32::MAX`).
    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }

    /// Returns `true` if this entity ID is **not** the null sentinel.
    #[inline]
    pub const fn is_valid(self) -> bool {
        !self.is_null()
    }

    /// Unwraps the raw `u32` representation.
    ///
    /// The lower 24 bits are the index and the upper 8 bits are the generation.
    #[inline]
    pub const fn into_raw(self) -> u32 {
        self.0
    }

    /// Constructs an `EntityId` from a raw `u32` value.
    ///
    /// # Safety
    ///
    /// The caller must ensure the value was produced by [`EntityId::into_raw`]
    /// (or an equivalent source) so that the bit layout is correct.
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        EntityId(raw)
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl fmt::Debug for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EntityId")
            .field(&format_args!("{:#010x}", self.0))
            .finish()
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            f.pad("Entity(NULL)")
        } else {
            write!(f, "Entity({}, gen={})", self.index(), self.generation())
        }
    }
}

impl Default for EntityId {
    #[inline]
    fn default() -> Self {
        EntityId::NULL
    }
}

impl From<EntityId> for u32 {
    #[inline]
    fn from(id: EntityId) -> Self {
        id.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_round_trips_index_and_generation() {
        let id = EntityId::new(42, 3);
        assert_eq!(id.index(), 42);
        assert_eq!(id.generation(), 3);
    }

    #[test]
    fn null_sentinel() {
        assert!(EntityId::NULL.is_null());
        assert!(!EntityId::NULL.is_valid());
    }

    #[test]
    fn max_entities_constant() {
        assert_eq!(MAX_ENTITIES, 16_777_215);
    }

    #[test]
    fn index_truncated_to_24_bits() {
        let id = EntityId::new(0xFFFFFFFF, 0);
        assert_eq!(id.index(), MAX_ENTITIES);
    }

    #[test]
    fn generation_truncated_to_8_bits() {
        let id = EntityId::new(0, 0xFF);
        assert_eq!(id.generation(), 0xFF);
    }

    #[test]
    fn bytemuck_pod() {
        let id = EntityId::new(1, 2);
        let bytes: &[u8] = bytemuck::bytes_of(&id);
        // Little-endian: index=1 → bytes[0]=1, generation=2 → bytes[3]=2
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[3], 2);
    }

    #[test]
    fn display_valid() {
        let id = EntityId::new(42, 3);
        assert_eq!(format!("{id}"), "Entity(42, gen=3)");
    }

    #[test]
    fn display_null() {
        assert_eq!(format!("{}", EntityId::NULL), "Entity(NULL)");
    }

    #[test]
    fn debug_format() {
        let id = EntityId::new(0, 0);
        let s = format!("{id:?}");
        assert!(s.starts_with("EntityId("));
    }

    #[test]
    fn default_is_null() {
        assert!(EntityId::default().is_null());
    }

    #[test]
    fn eq_and_hash() {
        let a = EntityId::new(7, 1);
        let b = EntityId::new(7, 1);
        let c = EntityId::new(7, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn raw_round_trip() {
        let id = EntityId::new(12345, 99);
        let raw = id.into_raw();
        let restored = EntityId::from_raw(raw);
        assert_eq!(id, restored);
    }

    #[test]
    fn from_entity_id_to_u32() {
        let id = EntityId::new(0, 0);
        let raw: u32 = id.into();
        assert_eq!(raw, 0);
    }
}
