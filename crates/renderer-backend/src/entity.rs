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
    use std::collections::HashSet;

    // =========================================================================
    // Entity Creation Tests (10+)
    // =========================================================================

    #[test]
    fn new_round_trips_index_and_generation() {
        let id = EntityId::new(42, 3);
        assert_eq!(id.index(), 42);
        assert_eq!(id.generation(), 3);
    }

    #[test]
    fn new_with_zero_index_and_zero_generation() {
        let id = EntityId::new(0, 0);
        assert_eq!(id.index(), 0);
        assert_eq!(id.generation(), 0);
        assert!(id.is_valid());
    }

    #[test]
    fn new_with_max_valid_index() {
        let id = EntityId::new(MAX_ENTITIES, 0);
        assert_eq!(id.index(), MAX_ENTITIES);
        assert_eq!(id.generation(), 0);
    }

    #[test]
    fn new_with_max_generation() {
        let id = EntityId::new(0, 255);
        assert_eq!(id.index(), 0);
        assert_eq!(id.generation(), 255);
    }

    #[test]
    fn new_with_max_index_and_max_generation_is_null() {
        // EntityId::new(MAX_ENTITIES, 255) produces 0xFF_FFFFFF which equals u32::MAX
        // Therefore this combination is actually the NULL sentinel
        let id = EntityId::new(MAX_ENTITIES, 255);
        assert_eq!(id.index(), MAX_ENTITIES);
        assert_eq!(id.generation(), 255);
        assert!(id.is_null(), "max index + max generation should be NULL");
        assert_eq!(id, EntityId::NULL);
    }

    #[test]
    fn new_with_various_indices() {
        let test_cases: [(u32, u8); 8] = [
            (1, 0),
            (100, 1),
            (1000, 10),
            (10000, 50),
            (100000, 100),
            (1000000, 200),
            (MAX_ENTITIES / 2, 128),
            (MAX_ENTITIES - 1, 254),
        ];
        for (index, gen) in test_cases {
            let id = EntityId::new(index, gen);
            assert_eq!(id.index(), index, "index mismatch for ({}, {})", index, gen);
            assert_eq!(id.generation(), gen, "generation mismatch for ({}, {})", index, gen);
        }
    }

    #[test]
    fn new_is_const_fn() {
        // Verify EntityId::new can be used in const context
        const ID: EntityId = EntityId::new(123, 45);
        assert_eq!(ID.index(), 123);
        assert_eq!(ID.generation(), 45);
    }

    #[test]
    fn null_const_is_const() {
        const NULL: EntityId = EntityId::NULL;
        assert!(NULL.is_null());
    }

    #[test]
    fn clone_produces_identical_entity() {
        let original = EntityId::new(999, 88);
        let cloned = original.clone();
        assert_eq!(original, cloned);
        assert_eq!(original.index(), cloned.index());
        assert_eq!(original.generation(), cloned.generation());
    }

    #[test]
    fn copy_semantics() {
        let id1 = EntityId::new(50, 5);
        let id2 = id1; // Copy, not move
        assert_eq!(id1, id2);
        // Both should still be usable (Copy trait)
        assert_eq!(id1.index(), 50);
        assert_eq!(id2.index(), 50);
    }

    // =========================================================================
    // Null Sentinel Tests (5+)
    // =========================================================================

    #[test]
    fn null_sentinel() {
        assert!(EntityId::NULL.is_null());
        assert!(!EntityId::NULL.is_valid());
    }

    #[test]
    fn null_raw_value_is_u32_max() {
        assert_eq!(EntityId::NULL.into_raw(), u32::MAX);
    }

    #[test]
    fn null_from_raw_u32_max() {
        let id = EntityId::from_raw(u32::MAX);
        assert!(id.is_null());
        assert_eq!(id, EntityId::NULL);
    }

    #[test]
    fn null_index_and_generation() {
        // NULL has all bits set, so index = 0xFFFFFF and generation = 0xFF
        assert_eq!(EntityId::NULL.index(), MAX_ENTITIES);
        assert_eq!(EntityId::NULL.generation(), 255);
    }

    #[test]
    fn max_bits_combination_equals_null() {
        // EntityId::new(MAX_ENTITIES, 255) produces exactly u32::MAX
        // because: (255 << 24) | 0xFFFFFF = 0xFF_000000 | 0x00_FFFFFF = 0xFFFFFFFF
        let id = EntityId::new(MAX_ENTITIES, 255);
        assert_eq!(id.into_raw(), u32::MAX);
        assert!(id.is_null());
        assert_eq!(id, EntityId::NULL);
    }

    #[test]
    fn near_max_values_is_valid() {
        // One below max index with max generation should NOT be null
        let id = EntityId::new(MAX_ENTITIES - 1, 255);
        assert!(id.is_valid());
        assert!(!id.is_null());

        // Max index with one below max generation should NOT be null
        let id2 = EntityId::new(MAX_ENTITIES, 254);
        assert!(id2.is_valid());
        assert!(!id2.is_null());
    }

    // =========================================================================
    // Boundary and Truncation Tests (10+)
    // =========================================================================

    #[test]
    fn max_entities_constant() {
        assert_eq!(MAX_ENTITIES, 16_777_215);
        assert_eq!(MAX_ENTITIES, (1 << 24) - 1);
        assert_eq!(MAX_ENTITIES, 0xFFFFFF);
    }

    #[test]
    fn index_truncated_to_24_bits() {
        let id = EntityId::new(0xFFFFFFFF, 0);
        assert_eq!(id.index(), MAX_ENTITIES);
    }

    #[test]
    fn index_truncation_discards_high_bits() {
        // Index 0x01_000000 should become 0 after truncation
        let id = EntityId::new(0x01_000000, 0);
        assert_eq!(id.index(), 0);
    }

    #[test]
    fn index_truncation_preserves_low_bits() {
        // Index 0x01_000005 should become 5 after truncation
        let id = EntityId::new(0x01_000005, 0);
        assert_eq!(id.index(), 5);
    }

    #[test]
    fn generation_truncated_to_8_bits() {
        let id = EntityId::new(0, 0xFF);
        assert_eq!(id.generation(), 0xFF);
    }

    #[test]
    fn generation_high_bits_discarded() {
        // Passing a value > 255 - it gets cast to u8, effectively truncating
        // Since generation param is u8, the compiler handles this
        let gen: u8 = 0xFF;
        let id = EntityId::new(0, gen);
        assert_eq!(id.generation(), 255);
    }

    #[test]
    fn bit_packing_index_low_24_bits() {
        let id = EntityId::new(0xABCDEF, 0);
        assert_eq!(id.into_raw() & INDEX_MASK, 0xABCDEF);
    }

    #[test]
    fn bit_packing_generation_high_8_bits() {
        let id = EntityId::new(0, 0xAB);
        assert_eq!(id.into_raw() >> 24, 0xAB);
    }

    #[test]
    fn bit_packing_combined() {
        let id = EntityId::new(0x123456, 0x78);
        let raw = id.into_raw();
        assert_eq!(raw, 0x78_123456);
    }

    #[test]
    fn index_and_generation_do_not_overlap() {
        // Ensure setting max index doesn't affect generation
        let id = EntityId::new(MAX_ENTITIES, 0);
        assert_eq!(id.generation(), 0);

        // Ensure setting max generation doesn't affect index
        let id2 = EntityId::new(0, 255);
        assert_eq!(id2.index(), 0);
    }

    #[test]
    fn sequential_indices_produce_sequential_raw_values() {
        let id0 = EntityId::new(0, 0);
        let id1 = EntityId::new(1, 0);
        let id2 = EntityId::new(2, 0);
        assert_eq!(id1.into_raw() - id0.into_raw(), 1);
        assert_eq!(id2.into_raw() - id1.into_raw(), 1);
    }

    // =========================================================================
    // Bytemuck / Pod / Zeroable Tests (5+)
    // =========================================================================

    #[test]
    fn bytemuck_pod() {
        let id = EntityId::new(1, 2);
        let bytes: &[u8] = bytemuck::bytes_of(&id);
        // Little-endian: index=1 -> bytes[0]=1, generation=2 -> bytes[3]=2
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[3], 2);
    }

    #[test]
    fn bytemuck_zeroable() {
        let zeroed: EntityId = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.index(), 0);
        assert_eq!(zeroed.generation(), 0);
        assert_eq!(zeroed.into_raw(), 0);
    }

    #[test]
    fn bytemuck_cast_from_u32() {
        let raw: u32 = 0x12_345678;
        let id: EntityId = bytemuck::cast(raw);
        assert_eq!(id.into_raw(), raw);
        assert_eq!(id.index(), 0x345678);
        assert_eq!(id.generation(), 0x12);
    }

    #[test]
    fn bytemuck_cast_to_u32() {
        let id = EntityId::new(0xABCDEF, 0x12);
        let raw: u32 = bytemuck::cast(id);
        assert_eq!(raw, 0x12_ABCDEF);
    }

    #[test]
    fn bytemuck_bytes_roundtrip() {
        let original = EntityId::new(0x123456, 0x78);
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let restored: &EntityId = bytemuck::from_bytes(bytes);
        assert_eq!(*restored, original);
    }

    #[test]
    fn bytemuck_slice_cast() {
        let ids = [EntityId::new(1, 0), EntityId::new(2, 0), EntityId::new(3, 0)];
        let raw_slice: &[u32] = bytemuck::cast_slice(&ids);
        assert_eq!(raw_slice.len(), 3);
        assert_eq!(raw_slice[0], 1);
        assert_eq!(raw_slice[1], 2);
        assert_eq!(raw_slice[2], 3);
    }

    // =========================================================================
    // Display and Debug Formatting Tests (5+)
    // =========================================================================

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
    fn display_zero_index_zero_gen() {
        let id = EntityId::new(0, 0);
        assert_eq!(format!("{id}"), "Entity(0, gen=0)");
    }

    #[test]
    fn display_max_values_shows_null() {
        // MAX_ENTITIES with gen=255 is actually NULL
        let id = EntityId::new(MAX_ENTITIES, 255);
        assert_eq!(format!("{id}"), "Entity(NULL)");
    }

    #[test]
    fn display_near_max_values() {
        // One below max should show normally
        let id = EntityId::new(MAX_ENTITIES - 1, 255);
        assert_eq!(format!("{id}"), "Entity(16777214, gen=255)");
    }

    #[test]
    fn display_with_padding() {
        let id = EntityId::new(5, 1);
        let formatted = format!("{:>20}", id);
        // "Entity(5, gen=1)" is 16 chars, padded to 20
        assert!(formatted.len() >= 16);
        assert!(formatted.contains("Entity(5, gen=1)"));
    }

    #[test]
    fn debug_format() {
        let id = EntityId::new(0, 0);
        let s = format!("{id:?}");
        assert!(s.starts_with("EntityId("));
    }

    #[test]
    fn debug_format_contains_hex() {
        let id = EntityId::new(0xABC, 0xDE);
        let s = format!("{id:?}");
        // Debug shows hex representation
        assert!(s.contains("0x"));
    }

    #[test]
    fn debug_format_null() {
        let s = format!("{:?}", EntityId::NULL);
        assert!(s.contains("EntityId("));
        assert!(s.contains("0xffffffff"));
    }

    // =========================================================================
    // Default Trait Tests (3+)
    // =========================================================================

    #[test]
    fn default_is_null() {
        assert!(EntityId::default().is_null());
    }

    #[test]
    fn default_equals_null_constant() {
        assert_eq!(EntityId::default(), EntityId::NULL);
    }

    #[test]
    fn default_raw_is_u32_max() {
        assert_eq!(EntityId::default().into_raw(), u32::MAX);
    }

    // =========================================================================
    // Equality and Hash Tests (5+)
    // =========================================================================

    #[test]
    fn eq_and_hash() {
        let a = EntityId::new(7, 1);
        let b = EntityId::new(7, 1);
        let c = EntityId::new(7, 2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn eq_same_index_different_generation() {
        let a = EntityId::new(100, 0);
        let b = EntityId::new(100, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn eq_different_index_same_generation() {
        let a = EntityId::new(100, 5);
        let b = EntityId::new(101, 5);
        assert_ne!(a, b);
    }

    #[test]
    fn hash_consistent_with_eq() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn hash_entity(id: EntityId) -> u64 {
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            hasher.finish()
        }

        let a = EntityId::new(42, 7);
        let b = EntityId::new(42, 7);
        assert_eq!(hash_entity(a), hash_entity(b));
    }

    #[test]
    fn hash_in_hashset() {
        let mut set = HashSet::new();
        set.insert(EntityId::new(1, 0));
        set.insert(EntityId::new(2, 0));
        set.insert(EntityId::new(1, 0)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn hash_different_entities_likely_different() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        fn hash_entity(id: EntityId) -> u64 {
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            hasher.finish()
        }

        let a = EntityId::new(1, 0);
        let b = EntityId::new(2, 0);
        // Not guaranteed, but very likely
        assert_ne!(hash_entity(a), hash_entity(b));
    }

    // =========================================================================
    // Raw Conversion Tests (5+)
    // =========================================================================

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

    #[test]
    fn from_raw_zero() {
        let id = EntityId::from_raw(0);
        assert_eq!(id.index(), 0);
        assert_eq!(id.generation(), 0);
        assert!(id.is_valid());
    }

    #[test]
    fn from_raw_preserves_all_bits() {
        let raw: u32 = 0xDEADBEEF;
        let id = EntityId::from_raw(raw);
        assert_eq!(id.into_raw(), raw);
    }

    #[test]
    fn into_raw_is_const() {
        const ID: EntityId = EntityId::new(100, 50);
        const RAW: u32 = ID.into_raw();
        assert_eq!(RAW, (50u32 << 24) | 100);
    }

    #[test]
    fn from_raw_is_const() {
        const RAW: u32 = 0x12345678;
        const ID: EntityId = EntityId::from_raw(RAW);
        assert_eq!(ID.into_raw(), RAW);
    }

    // =========================================================================
    // is_valid / is_null Tests (5+)
    // =========================================================================

    #[test]
    fn is_valid_for_zero() {
        let id = EntityId::new(0, 0);
        assert!(id.is_valid());
        assert!(!id.is_null());
    }

    #[test]
    fn is_valid_for_near_max_values() {
        // MAX_ENTITIES with gen=255 is NULL, so test near-max instead
        let id = EntityId::new(MAX_ENTITIES - 1, 255);
        assert!(id.is_valid());

        let id2 = EntityId::new(MAX_ENTITIES, 254);
        assert!(id2.is_valid());
    }

    #[test]
    fn is_null_only_for_null_sentinel() {
        // Test many values - only the NULL combination should be null
        let test_ids = [
            EntityId::new(0, 0),
            EntityId::new(1, 0),
            EntityId::new(MAX_ENTITIES, 0),
            EntityId::new(0, 255),
            EntityId::new(MAX_ENTITIES - 1, 255),  // Near-max, not null
            EntityId::new(MAX_ENTITIES, 254),      // Near-max, not null
        ];
        for id in test_ids {
            assert!(!id.is_null(), "Unexpected null for {:?}", id);
        }

        // Only this specific combination should be null
        let null_combo = EntityId::new(MAX_ENTITIES, 255);
        assert!(null_combo.is_null(), "MAX_ENTITIES + gen=255 should be NULL");
    }

    #[test]
    fn is_null_is_const() {
        const IS_NULL: bool = EntityId::NULL.is_null();
        const IS_VALID: bool = EntityId::NULL.is_valid();
        assert!(IS_NULL);
        assert!(!IS_VALID);
    }

    #[test]
    fn is_valid_is_const() {
        const ID: EntityId = EntityId::new(42, 1);
        const VALID: bool = ID.is_valid();
        const NULL: bool = ID.is_null();
        assert!(VALID);
        assert!(!NULL);
    }

    // =========================================================================
    // Index and Generation Accessor Tests (5+)
    // =========================================================================

    #[test]
    fn index_is_const() {
        const ID: EntityId = EntityId::new(999, 10);
        const IDX: u32 = ID.index();
        assert_eq!(IDX, 999);
    }

    #[test]
    fn generation_is_const() {
        const ID: EntityId = EntityId::new(999, 10);
        const GEN: u8 = ID.generation();
        assert_eq!(GEN, 10);
    }

    #[test]
    fn index_zero() {
        assert_eq!(EntityId::new(0, 100).index(), 0);
    }

    #[test]
    fn generation_zero() {
        assert_eq!(EntityId::new(100, 0).generation(), 0);
    }

    #[test]
    fn index_boundary_values() {
        // Test power-of-two boundaries
        for i in 0..24 {
            let idx = 1u32 << i;
            let id = EntityId::new(idx, 0);
            assert_eq!(id.index(), idx, "Failed at 2^{}", i);
        }
    }

    // =========================================================================
    // repr(transparent) Tests (3+)
    // =========================================================================

    #[test]
    fn size_of_entity_id_is_u32() {
        assert_eq!(std::mem::size_of::<EntityId>(), std::mem::size_of::<u32>());
    }

    #[test]
    fn align_of_entity_id_is_u32() {
        assert_eq!(std::mem::align_of::<EntityId>(), std::mem::align_of::<u32>());
    }

    #[test]
    fn transmute_from_u32_matches_from_raw() {
        let raw: u32 = 0x12345678;
        // Safety: EntityId is repr(transparent) over u32
        let transmuted: EntityId = unsafe { std::mem::transmute(raw) };
        let from_raw = EntityId::from_raw(raw);
        assert_eq!(transmuted, from_raw);
    }

    // =========================================================================
    // Edge Case and Stress Tests (5+)
    // =========================================================================

    #[test]
    fn many_entities_unique() {
        let mut set = HashSet::new();
        for i in 0..1000 {
            let id = EntityId::new(i, (i % 256) as u8);
            assert!(set.insert(id), "Duplicate entity at index {}", i);
        }
        assert_eq!(set.len(), 1000);
    }

    #[test]
    fn generation_wraparound_simulation() {
        // Simulate entity slot reuse with generation increment
        let mut gen: u8 = 0;
        let index = 42;
        let mut prev_id = EntityId::new(index, gen);

        for _ in 0..512 {
            gen = gen.wrapping_add(1);
            let new_id = EntityId::new(index, gen);
            assert_ne!(prev_id, new_id, "Stale handle detection should work");
            prev_id = new_id;
        }
    }

    #[test]
    fn null_not_equal_to_any_valid_entity() {
        for i in 0..100 {
            let id = EntityId::new(i, i as u8);
            assert_ne!(id, EntityId::NULL);
        }
    }

    #[test]
    fn consecutive_indices_have_different_hashes() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hashes = HashSet::new();
        for i in 0..100 {
            let id = EntityId::new(i, 0);
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            hashes.insert(hasher.finish());
        }
        // All 100 should be unique
        assert_eq!(hashes.len(), 100);
    }

    #[test]
    fn entity_in_vec() {
        let mut entities: Vec<EntityId> = Vec::new();
        for i in 0..100 {
            entities.push(EntityId::new(i, 0));
        }
        assert_eq!(entities.len(), 100);
        assert_eq!(entities[50].index(), 50);
    }

    #[test]
    fn entity_as_option() {
        let some_entity: Option<EntityId> = Some(EntityId::new(1, 0));
        let no_entity: Option<EntityId> = None;

        assert!(some_entity.is_some());
        assert!(no_entity.is_none());
        assert_eq!(some_entity.unwrap().index(), 1);
    }

    #[test]
    fn entity_comparison_ordering() {
        // EntityId doesn't implement Ord, but PartialEq works
        let a = EntityId::new(1, 0);
        let b = EntityId::new(1, 0);
        let c = EntityId::new(2, 0);

        assert!(a == b);
        assert!(a != c);
        assert!(b != c);
    }
}
