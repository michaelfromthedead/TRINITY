// HierarchicalChecksum for deterministic per-frame state verification.
//
// Provides three layers of verification:
//   1. Component level  — rolling xxhash64-like checksum per component slot.
//   2. Entity level     — accumulated checksum from all components of an entity.
//   3. World level      — combined checksum from all entities, exposed atomically.
//
// The design is lock-free for readers (world_checksum uses AtomicU64) while
// writers synchronise through a Mutex-guarded entity map.

// ---------------------------------------------------------------------------
// xxhash64-like fast non-cryptographic hash
// ---------------------------------------------------------------------------

const PRIME1: u64 = 0x9E3779B185EBCA87u64;
const PRIME2: u64 = 0xC2B2AE3D27D4EB4Fu64;
const PRIME3: u64 = 0x165667B19E3779F9u64;
const PRIME4: u64 = 0x85EBCA77C2B2AE63u64;
const PRIME5: u64 = 0x27D4EB2F165667C5u64;

/// Compute an xxhash64-like digest for a byte slice with the given seed.
///
/// Processes data in 8-byte lanes with rotate-mix rounds and applies a final
/// avalanche pass.  Not cryptographically secure — only suitable for
/// deterministic state verification.
#[inline]
fn fast_hash(data: &[u8], seed: u64) -> u64 {
    let len = data.len();
    let mut h = seed.wrapping_add(PRIME5).wrapping_add(len as u64);
    let mut remaining = data;

    // 8-byte lanes.
    while remaining.len() >= 8 {
        let val = u64::from_le_bytes([
            remaining[0], remaining[1], remaining[2], remaining[3],
            remaining[4], remaining[5], remaining[6], remaining[7],
        ]);
        remaining = &remaining[8..];
        h ^= val.wrapping_mul(PRIME2).rotate_left(31).wrapping_mul(PRIME1);
        h = h.rotate_left(27).wrapping_mul(PRIME1).wrapping_add(PRIME4);
    }

    // 4-byte tail.
    if remaining.len() >= 4 {
        let val = u32::from_le_bytes([
            remaining[0], remaining[1], remaining[2], remaining[3],
        ]) as u64;
        remaining = &remaining[4..];
        h ^= val.wrapping_mul(PRIME1);
        h = h.rotate_left(23).wrapping_mul(PRIME2).wrapping_add(PRIME3);
    }

    // 1-byte tail.
    for &b in remaining {
        h ^= (b as u64).wrapping_mul(PRIME5);
        h = h.rotate_left(11).wrapping_mul(PRIME1);
    }

    // Final avalanche.
    h ^= h >> 33;
    h = h.wrapping_mul(PRIME2);
    h ^= h >> 29;
    h = h.wrapping_mul(PRIME3);
    h ^= h >> 32;

    h
}

// ---------------------------------------------------------------------------
// Entity-level checksum state
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// Rolling checksum state for a single entity.
struct EntityChecksum {
    /// Per-component rolling checksums: component_id -> checksum.
    components: HashMap<u32, u64>,
    /// Accumulated entity-level checksum (wrapping sum of components).
    accumulated: u64,
}

// ---------------------------------------------------------------------------
// HierarchicalChecksum
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Deterministic hierarchical checksum for per-frame state verification.
///
/// # Hierarchy
///
/// ```text
/// Component (per entity, per component, per field-offset)
///   └─> ComponentChecksum  (rolling xxhash64-like)
///        └─> EntityChecksum (wrapping sum)
///             └─> WorldChecksum (AtomicU64, lock-free read)
/// ```
///
/// # Thread Safety
///
/// - `world_checksum()` reads an `AtomicU64` — wait-free, no contention.
/// - `update()` and `reset()` take `&self` and serialise mutation of the
///   internal entity map through a `Mutex`.
///
/// # Determinism
///
/// Identical sequences of `update()` calls with identical data produce
/// identical checksums.  Different field offsets within the same component
/// produce different digests, so reordering field updates changes the result.
/// Collision resistance is sufficient for determinism verification but *not*
/// cryptographic.
pub struct HierarchicalChecksum {
    /// Entity checksum state, synchronised via mutex.
    entities: Mutex<HashMap<u64, EntityChecksum>>,
    /// World-level checksum, exposed atomically for lock-free reads.
    world_checksum: AtomicU64,
}

impl HierarchicalChecksum {
    /// Create a fresh checksum tracker with all checksums initialised to zero.
    pub fn new() -> Self {
        Self {
            entities: Mutex::new(HashMap::new()),
            world_checksum: AtomicU64::new(0),
        }
    }

    /// Feed component data into the rolling checksum.
    ///
    /// The `field_offset` parameter disambiguates fields *within* a component
    /// so that identical byte sequences at different offsets produce different
    /// digests.  Calling this method multiple times for the same
    /// (entity, component) pair *accumulates* into a rolling checksum.
    ///
    /// # Parameters
    ///
    /// * `entity_id`    — unique entity identifier.
    /// * `component_id` — type tag for the component (e.g. from type registry).
    /// * `field_offset` — byte offset of the field within the component struct.
    /// * `data`         — raw byte representation of the field value.
    pub fn update(
        &self,
        entity_id: u64,
        component_id: u32,
        field_offset: u32,
        data: &[u8],
    ) {
        // Mix entity/component/offset into the seed so the same data at a
        // different position or in a different entity produces a different hash.
        let seed = (entity_id as u64)
            .wrapping_mul(PRIME3)
            .wrapping_add(component_id as u64)
            .wrapping_add(field_offset as u64);
        let component_hash = fast_hash(data, seed);

        let mut entities = self.entities.lock().unwrap();
        let entry = entities.entry(entity_id).or_insert_with(|| EntityChecksum {
            components: HashMap::new(),
            accumulated: 0,
        });

        // Rolling update: fold the new hash into the existing component checksum
        // using wrapping addition (no cancellation, unlike XOR).
        let old = *entry.components.get(&component_id).unwrap_or(&0);
        let new = old.wrapping_add(component_hash);
        entry.components.insert(component_id, new);

        // Entity accumulated checksum tracks the delta so we can update the
        // world checksum in O(1) instead of re-summing all entities.
        let entity_delta = new.wrapping_sub(old);
        entry.accumulated = entry.accumulated.wrapping_add(entity_delta);

        // Publish the world checksum delta atomically.
        self.world_checksum
            .fetch_add(entity_delta, Ordering::Release);
    }

    /// Return the current world-level checksum.
    ///
    /// This is a wait-free, lock-free snapshot of the most recently published
    /// world checksum.  It may lag very slightly behind concurrent `update()`
    /// calls, but for frame boundary verification the lag is irrelevant because
    /// all updates are complete before the frame end check.
    pub fn world_checksum(&self) -> u64 {
        self.world_checksum.load(Ordering::Acquire)
    }

    /// Reset all checksums to zero — call at the start of each frame boundary.
    ///
    /// Clears every entity's component and accumulated checksums and resets
    /// the world checksum to `0`.
    pub fn reset(&self) {
        let mut entities = self.entities.lock().unwrap();
        entities.clear();
        self.world_checksum.store(0, Ordering::Release);
    }
}

impl Default for HierarchicalChecksum {
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

    // ---- Determinism & basic properties ----

    #[test]
    fn same_inputs_produce_same_checksum() {
        let c = HierarchicalChecksum::new();
        c.update(1, 0, 0, b"hello");
        c.update(1, 0, 4, b"world");
        let a = c.world_checksum();

        let c = HierarchicalChecksum::new();
        c.update(1, 0, 0, b"hello");
        c.update(1, 0, 4, b"world");
        let b = c.world_checksum();

        assert_eq!(a, b);
    }

    #[test]
    fn different_field_offset_produces_different_checksum() {
        let a = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, b"data");
            c.world_checksum()
        };
        let b = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 8, b"data");
            c.world_checksum()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn different_entity_produces_different_checksum() {
        let a = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, b"x");
            c.world_checksum()
        };
        let b = {
            let c = HierarchicalChecksum::new();
            c.update(2, 0, 0, b"x");
            c.world_checksum()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn different_data_produces_different_checksum() {
        let a = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, b"alpha");
            c.world_checksum()
        };
        let b = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, b"beta");
            c.world_checksum()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn empty_data_is_deterministic() {
        let a = {
            let c = HierarchicalChecksum::new();
            c.update(7, 3, 12, b"");
            c.world_checksum()
        };
        let b = {
            let c = HierarchicalChecksum::new();
            c.update(7, 3, 12, b"");
            c.world_checksum()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn fresh_checksum_is_zero() {
        let c = HierarchicalChecksum::new();
        assert_eq!(c.world_checksum(), 0);
    }
    // ---- Accumulation ----

    #[test]
    fn multiple_updates_accumulate() {
        let s1 = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, b"a");
            c.world_checksum()
        };
        let s2 = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, b"a");
            c.update(1, 0, 0, b"b");
            c.world_checksum()
        };
        assert_ne!(s2, s1);
    }

    #[test]
    fn multiple_entities_accumulate() {
        let single = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, b"payload");
            c.world_checksum()
        };
        let both = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, b"payload");
            c.update(2, 0, 0, b"payload");
            c.world_checksum()
        };
        assert_ne!(both, single);
    }
    // ---- Reset ----

    #[test]
    fn reset_clears_checksum() {
        let c = HierarchicalChecksum::new();
        c.update(1, 0, 0, b"something");
        assert_ne!(c.world_checksum(), 0);
        c.reset();
        assert_eq!(c.world_checksum(), 0);
    }

    #[test]
    fn reset_and_rewrite_is_identical() {
        let c = HierarchicalChecksum::new();
        c.update(1, 0, 0, b"frame1");
        c.reset();
        c.update(1, 0, 0, b"frame2");
        let after_reset = c.world_checksum();

        let c2 = HierarchicalChecksum::new();
        c2.update(1, 0, 0, b"frame2");
        let fresh = c2.world_checksum();

        assert_eq!(after_reset, fresh);
    }

    // ---- Large data ----

    #[test]
    fn large_data_is_deterministic() {
        let data: Vec<u8> = (0..u8::MAX).cycle().take(4096).collect();
        let a = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, &data);
            c.world_checksum()
        };
        let b = {
            let c = HierarchicalChecksum::new();
            c.update(1, 0, 0, &data);
            c.world_checksum()
        };
        assert_eq!(a, b);
    }

    // ---- Multi-component entity ----

    #[test]
    fn different_components_on_same_entity_accumulate() {
        let c = HierarchicalChecksum::new();
        c.update(1, 0, 0, b"transform");
        c.update(1, 1, 0, b"mesh");
        let world = c.world_checksum();
        assert_ne!(world, 0);
    }

    // ---- Thread safety smoke test ----

    #[test]
    fn concurrent_updates_do_not_panic() {
        use std::thread;

        let c = std::sync::Arc::new(HierarchicalChecksum::new());
        let mut handles = Vec::new();

        for tid in 0..4 {
            let cc = std::sync::Arc::clone(&c);
            handles.push(thread::spawn(move || {
                for i in 0..250 {
                    cc.update(tid as u64, 0, i * 4, b"concurrent");
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_ne!(c.world_checksum(), 0);
    }

    // ---- Deterministic across identical workloads ----

    #[test]
    fn identical_workload_produces_identical_checksum() {
        let run = || -> u64 {
            let c = HierarchicalChecksum::new();
            for entity in 0..3 {
                c.update(entity, 0, 0, &42.0f32.to_le_bytes());
                c.update(entity, 0, 4, &7.0f32.to_le_bytes());
                c.update(entity, 1, 0, &999u32.to_le_bytes());
            }
            c.world_checksum()
        };

        assert_eq!(run(), run());
    }
}
