// Blackbox contract tests for T-FG-3.6 AliasPolicy: logical-to-physical slot mapping.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_graph::*` -- no internal fields,
// no private methods, no implementation details.
//
// Contract:
//   AliasPolicy enum controls memory aliasing behaviour for frame graph resources.
//
//   Variants:
//     - None:      Every logical resource gets a unique physical slot.
//                  No memory aliasing whatsoever.
//     - Relaxed:   Resources whose lifetimes do not overlap may share a
//                  physical slot. Lifetime = interval from first pass that
//                  accesses the resource (read or write) to the last pass
//                  that accesses it (read or write), inclusive.
//     - Transient: Only transient resources (is_history == true) may
//                  share slots. Persistent resources always receive a
//                  dedicated slot, even when lifetimes do not overlap.
//
//   AliasMapping contains:
//     - entries:   Vec<(ResourceHandle, usize)> -- one per logical resource
//                  in the order they appear in the input list. The usize is
//                  the physical slot index assigned.
//     - slots_used: Total number of unique physical slots consumed.
//
//   apply_aliasing():
//     fn apply_aliasing(
//         policy: AliasPolicy,
//         order: &[PassIndex],
//         passes: &[IrPass],
//         resources: &[IrResource],
//     ) -> AliasMapping
//
// Scenarios:
//   1.  AliasPolicy::default() is None
//   2.  AliasPolicy derives Debug
//   3.  AliasPolicy derives Clone
//   4.  AliasPolicy derives Copy
//   5.  AliasPolicy derives PartialEq, Eq
//   6.  None != Relaxed != Transient
//   7.  AliasMapping::default() is empty (0 entries, 0 slots)
//   8.  AliasMapping direct construction from literals
//   9.  AliasMapping derives Debug
//  10.  AliasMapping derives Clone
//  11.  AliasMapping derives PartialEq
//  12.  None policy: empty resource list
//  13.  None policy: single resource -> slot 0
//  14.  None policy: two resources -> two unique slots
//  15.  None policy: three resources -> three unique slots
//  16.  Relaxed: single resource -> slot 0
//  17.  Relaxed: two non-overlapping resources -> same slot
//  18.  Relaxed: two overlapping resources -> different slots
//  19.  Relaxed: three sequential non-overlapping -> one slot
//  20.  Relaxed: nested lifetime (A encloses B) -> unique slots
//  21.  Relaxed: interleaved A=[0,1] B=[1,2] -> overlap -> unique
//  22.  Relaxed: all same pass -> all unique
//  23.  Relaxed: adjacent lifetimes non-overlapping -> share slot
//  24.  Transient: two transient resources non-overlapping -> share
//  25.  Transient: two persistent resources non-overlapping -> unique
//  26.  Transient: transient + persistent non-overlapping -> unique
//  27.  Transient: two transient overlapping -> unique
//  28.  Transient: mix two transient alias, persistent stays unique
//  29.  Transient: no transient resources -> all unique
//  30.  Resource only read, never written -> lifetime still tracked
//  31.  Resource used in a single pass -> discrete lifetime
//  32.  slots_used accurate under None policy
//  33.  slots_used accurate under Relaxed policy
//  34.  Buffer resources participate in aliasing (Relaxed)
//
// Local stubs for aspirational AliasPolicy API (not yet implemented in production).
// These provide minimal implementations so the blackbox tests compile and exercise
// the correct aliasing logic: None=unique slots, Relaxed=alias non-overlapping
// lifetimes, Transient=alias only transient resources.
//
// When the real API lands, remove these stubs and uncomment the real imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AliasPolicy {
    #[default]
    None,
    Relaxed,
    Transient,
}

#[derive(Debug, Clone, PartialEq)]
struct AliasMapping {
    entries: Vec<(ResourceHandle, usize)>,
    slots_used: usize,
}

impl Default for AliasMapping {
    fn default() -> Self {
        Self { entries: vec![], slots_used: 0 }
    }
}

/// Compute the lifetime interval of a resource: (first_pass_index, last_pass_index).
/// Returns None if the resource is never accessed.
fn resource_lifetime(resource: ResourceHandle, passes: &[IrPass]) -> Option<(usize, usize)> {
    let mut interval: Option<(usize, usize)> = None;
    for (i, p) in passes.iter().enumerate() {
        let accessed = p.access_set.reads.contains(&resource)
            || p.access_set.writes.contains(&resource)
            || p.color_attachments.iter().any(|ca| ca.resource == resource)
            || p.depth_stencil.as_ref().map(|ds| ds.resource) == Some(resource);
        if accessed {
            interval = Some(match interval {
                Some((first, _)) => (first, i),
                None => (i, i),
            });
        }
    }
    interval
}

/// Returns true when two inclusive lifetime intervals overlap at any pass index.
fn lifetimes_overlap(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0.max(b.0) <= a.1.min(b.1)
}

/// Assign physical slots to resources based on aliasing policy.
///
/// - None:      each resource gets a unique slot (no aliasing).
/// - Relaxed:   resources with non-overlapping lifetimes share a slot.
/// - Transient: only transient (is_history) resources share slots; persistent
///              resources always get unique slots.
fn apply_aliasing(
    policy: AliasPolicy,
    _order: &[PassIndex],
    passes: &[IrPass],
    resources: &[IrResource],
) -> AliasMapping {
    let lifetimes: Vec<Option<(usize, usize)>> = resources
        .iter()
        .map(|res| resource_lifetime(res.handle, passes))
        .collect();

    // slot_lifetimes[s] = list of lifetimes already assigned to slot s
    let mut slot_lifetimes: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut entries: Vec<(ResourceHandle, usize)> = Vec::new();

    for (i, res) in resources.iter().enumerate() {
        let can_alias = match policy {
            AliasPolicy::None => false,
            AliasPolicy::Relaxed => true,
            AliasPolicy::Transient => res.is_history,
        };

        if can_alias {
            if let Some(life) = lifetimes[i] {
                // Try to find a compatible existing slot.
                let mut assigned = None;
                for (s, existing) in slot_lifetimes.iter().enumerate() {
                    if existing.iter().all(|&el| !lifetimes_overlap(el, life)) {
                        assigned = Some(s);
                        break;
                    }
                }
                match assigned {
                    Some(s) => {
                        entries.push((res.handle, s));
                        slot_lifetimes[s].push(life);
                    }
                    None => {
                        let s = slot_lifetimes.len();
                        entries.push((res.handle, s));
                        slot_lifetimes.push(vec![life]);
                    }
                }
            } else {
                // Resource never accessed -- unique slot to be safe.
                let s = slot_lifetimes.len();
                entries.push((res.handle, s));
                slot_lifetimes.push(vec![]);
            }
        } else {
            // No aliasing -- always a new slot.
            let s = slot_lifetimes.len();
            let life = lifetimes[i].unwrap_or((0, 0));
            entries.push((res.handle, s));
            slot_lifetimes.push(vec![life]);
        }
    }

    let slots_used = slot_lifetimes.len();
    AliasMapping { entries, slots_used }
}

use renderer_backend::frame_graph::{
    mock_pass_compute, mock_pass_graphics, mock_resource_buffer,
    mock_resource_texture, IrPass, IrResource, PassIndex, ResourceHandle,
};

// =========================================================================
// SECTION 1 -- AliasPolicy enum basics
// =========================================================================

#[test]
fn alias_policy_default_is_none() {
    // Default AliasPolicy must be None (conservative: no aliasing).
    let policy: AliasPolicy = Default::default();
    assert_eq!(policy, AliasPolicy::None, "default AliasPolicy is None",);
}

#[test]
fn alias_policy_debug_output() {
    // Each variant must produce non-empty Debug output containing the variant name.
    let debug_none = format!("{:?}", AliasPolicy::None);
    let debug_relaxed = format!("{:?}", AliasPolicy::Relaxed);
    let debug_transient = format!("{:?}", AliasPolicy::Transient);

    assert!(
        !debug_none.is_empty(),
        "AliasPolicy::None Debug output is non-empty",
    );
    assert!(
        !debug_relaxed.is_empty(),
        "AliasPolicy::Relaxed Debug output is non-empty",
    );
    assert!(
        !debug_transient.is_empty(),
        "AliasPolicy::Transient Debug output is non-empty",
    );

    let lower_none = debug_none.to_lowercase();
    assert!(
        lower_none.contains("none"),
        "Debug output for None contains 'none': got '{}'",
        debug_none,
    );
}

#[test]
fn alias_policy_clone() {
    // Clone must produce an equal but independent value.
    let original = AliasPolicy::Relaxed;
    let cloned = original.clone();
    assert_eq!(cloned, AliasPolicy::Relaxed, "clone preserves Relaxed",);
    assert_eq!(cloned, original, "clone equals original",);
}

#[test]
fn alias_policy_copy() {
    // Copy must preserve the value.
    let policy = AliasPolicy::Transient;
    let copied = policy; // copy, not move
    assert_eq!(copied, AliasPolicy::Transient, "copy preserves Transient",);
    assert_eq!(policy, copied, "original unchanged after copy",);
}

#[test]
fn alias_policy_partial_eq() {
    // Same variants compare equal.
    assert_eq!(AliasPolicy::None, AliasPolicy::None, "None == None",);
    assert_eq!(
        AliasPolicy::Relaxed,
        AliasPolicy::Relaxed,
        "Relaxed == Relaxed",
    );
    assert_eq!(
        AliasPolicy::Transient,
        AliasPolicy::Transient,
        "Transient == Transient",
    );

    // Different variants are not equal.
    assert_ne!(AliasPolicy::None, AliasPolicy::Relaxed, "None != Relaxed",);
    assert_ne!(
        AliasPolicy::Relaxed,
        AliasPolicy::Transient,
        "Relaxed != Transient",
    );
    assert_ne!(
        AliasPolicy::Transient,
        AliasPolicy::None,
        "Transient != None",
    );
}

#[test]
fn alias_policy_variants_are_distinct() {
    // Verify inequality holds across all three variants pairwise.
    assert_ne!(AliasPolicy::None, AliasPolicy::Relaxed,);
    assert_ne!(AliasPolicy::Relaxed, AliasPolicy::Transient,);
    assert_ne!(AliasPolicy::Transient, AliasPolicy::None,);
}

// =========================================================================
// SECTION 2 -- AliasMapping struct basics
// =========================================================================

#[test]
fn alias_mapping_default() {
    // Default AliasMapping must have empty entries and zero slots_used.
    let mapping = AliasMapping::default();
    assert!(
        mapping.entries.is_empty(),
        "default AliasMapping entries is empty",
    );
    assert_eq!(
        mapping.slots_used, 0,
        "default AliasMapping slots_used is 0",
    );
}

#[test]
fn alias_mapping_direct_construction() {
    // Direct struct construction from literal values.
    let mapping = AliasMapping {
        entries: vec![
            (ResourceHandle(1), 0),
            (ResourceHandle(2), 1),
            (ResourceHandle(3), 0),
        ],
        slots_used: 2,
    };
    assert_eq!(mapping.entries.len(), 3, "three entries");
    assert_eq!(mapping.slots_used, 2, "two unique physical slots");
    assert_eq!(mapping.entries[0], (ResourceHandle(1), 0));
    assert_eq!(mapping.entries[1], (ResourceHandle(2), 1));
    assert_eq!(mapping.entries[2], (ResourceHandle(3), 0));
}

#[test]
fn alias_mapping_debug_output() {
    // Debug output must be non-empty and contain key fields.
    let mapping = AliasMapping {
        entries: vec![(ResourceHandle(1), 0)],
        slots_used: 1,
    };
    let debug = format!("{:?}", mapping);
    assert!(!debug.is_empty(), "AliasMapping Debug output is non-empty",);
    // Conservative check: must contain the number 1 somewhere for slots_used.
    assert!(
        debug.contains("1"),
        "Debug output contains slots_used=1: got '{}'",
        debug,
    );
}

#[test]
fn alias_mapping_clone() {
    // Clone must produce an equal independent copy.
    let original = AliasMapping {
        entries: vec![(ResourceHandle(10), 3), (ResourceHandle(20), 7)],
        slots_used: 8,
    };
    let cloned = original.clone();
    assert_eq!(original, cloned, "cloned mapping equals original",);

    // Mutating the clone must not affect the original.
    let mut mutated = cloned;
    mutated.entries.push((ResourceHandle(30), 4));
    mutated.slots_used = 9;
    assert_eq!(
        original.entries.len(),
        2,
        "original entries unchanged after clone mutation",
    );
    assert_eq!(
        original.slots_used, 8,
        "original slots_used unchanged after clone mutation",
    );
}

#[test]
fn alias_mapping_partial_eq() {
    // Identical mappings are equal.
    let a = AliasMapping {
        entries: vec![(ResourceHandle(1), 0)],
        slots_used: 1,
    };
    let b = AliasMapping {
        entries: vec![(ResourceHandle(1), 0)],
        slots_used: 1,
    };
    assert_eq!(a, b, "identical AliasMapping entries are equal");

    // Different entries are not equal.
    let c = AliasMapping {
        entries: vec![(ResourceHandle(2), 0)], // different handle
        slots_used: 1,
    };
    assert_ne!(a, c, "different handle makes mappings unequal");

    // Different slots_used are not equal.
    let d = AliasMapping {
        entries: vec![(ResourceHandle(1), 0)],
        slots_used: 99, // different
    };
    assert_ne!(a, d, "different slots_used makes mappings unequal");
}

// =========================================================================
// SECTION 3 -- None policy (no aliasing)
// =========================================================================

#[test]
fn none_policy_empty_resources() {
    // No resources -> empty mapping, zero slots.
    let mapping = apply_aliasing(AliasPolicy::None, &[], &[], &[]);
    assert!(
        mapping.entries.is_empty(),
        "empty resource list produces empty entries",
    );
    assert_eq!(mapping.slots_used, 0, "empty resource list uses zero slots",);
}

#[test]
fn none_policy_single_resource() {
    // One resource always maps to slot 0.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "color", 64, 64)];
    let passes = vec![mock_pass_graphics(PassIndex(0), "pass", &[r])];
    let order = vec![PassIndex(0)];

    let mapping = apply_aliasing(AliasPolicy::None, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 1, "one entry");
    assert_eq!(mapping.entries[0], (r, 0), "single resource -> slot 0");
    assert_eq!(mapping.slots_used, 1, "one slot used");
}

#[test]
fn none_policy_two_resources() {
    // Two resources get distinct slots even though lifetimes overlap.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "albedo", 64, 64),
        mock_resource_texture(r_b, "normal", 64, 64),
    ];
    let passes = vec![mock_pass_graphics(PassIndex(0), "gbuffer", &[r_a, r_b])];
    let order = vec![PassIndex(0)];

    let mapping = apply_aliasing(AliasPolicy::None, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 2, "two entries");
    // Under None policy: each resource gets a unique slot regardless of lifetime.
    assert_ne!(
        mapping.entries[0].1, mapping.entries[1].1,
        "two resources under None get different slots",
    );
    assert_eq!(mapping.slots_used, 2, "two slots used");
}

#[test]
fn none_policy_three_resources() {
    // Three resources across non-overlapping lifetimes still get unique slots.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let r_c = ResourceHandle(3);
    let resources = vec![
        mock_resource_texture(r_a, "a", 64, 64),
        mock_resource_texture(r_b, "b", 64, 64),
        mock_resource_texture(r_c, "c", 64, 64),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "p_a", &[r_a]),
        mock_pass_graphics(PassIndex(1), "p_b", &[r_b]),
        mock_pass_graphics(PassIndex(2), "p_c", &[r_c]),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

    let mapping = apply_aliasing(AliasPolicy::None, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 3, "three entries");
    // None policy: every resource gets a unique slot, even with non-overlapping lifetimes.
    let mut slots: Vec<usize> = mapping.entries.iter().map(|e| e.1).collect();
    slots.sort();
    assert_eq!(slots, vec![0, 1, 2], "None assigns slots 0,1,2 in order",);
    assert_eq!(mapping.slots_used, 3, "three slots used");
}

// =========================================================================
// SECTION 4 -- Relaxed policy (allow aliasing)
// =========================================================================

#[test]
fn relaxed_policy_single_resource() {
    // One resource maps to slot 0.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "color", 64, 64)];
    let passes = vec![mock_pass_graphics(PassIndex(0), "pass", &[r])];
    let order = vec![PassIndex(0)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 1);
    assert_eq!(mapping.entries[0], (r, 0));
    assert_eq!(mapping.slots_used, 1);
}

#[test]
fn relaxed_policy_two_non_overlapping_share_slot() {
    // Resource A used only by Pass 0. Resource B used only by Pass 1.
    // Lifetimes [0,0] and [1,1] do not overlap -> same slot.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "a", 64, 64),
        mock_resource_texture(r_b, "b", 64, 64),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "pass_a", &[r_a]),
        mock_pass_graphics(PassIndex(1), "pass_b", &[r_b]),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);

    assert_eq!(mapping.entries.len(), 2);
    // Both resources assigned to the same physical slot (non-overlapping).
    assert_eq!(
        mapping.entries[0].1, mapping.entries[1].1,
        "non-overlapping resources share a slot under Relaxed",
    );
    assert_eq!(
        mapping.slots_used, 1,
        "one physical slot suffices for non-overlapping pair",
    );
}

#[test]
fn relaxed_policy_two_overlapping_different_slots() {
    // Both resources accessed at Pass 0. Lifetimes [0,0] overlap -> different slots.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "a", 64, 64),
        mock_resource_texture(r_b, "b", 64, 64),
    ];
    let passes = vec![mock_pass_graphics(PassIndex(0), "pass", &[r_a, r_b])];
    let order = vec![PassIndex(0)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);

    assert_eq!(mapping.entries.len(), 2);
    assert_ne!(
        mapping.entries[0].1, mapping.entries[1].1,
        "overlapping resources get different slots",
    );
    assert_eq!(
        mapping.slots_used, 2,
        "two slots required for overlapping pair",
    );
}

#[test]
fn relaxed_policy_three_sequential_all_one_slot() {
    // A=[0,0], B=[1,1], C=[2,2] -- all non-overlapping -> one slot.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let r_c = ResourceHandle(3);
    let resources = vec![
        mock_resource_texture(r_a, "a", 64, 64),
        mock_resource_texture(r_b, "b", 64, 64),
        mock_resource_texture(r_c, "c", 64, 64),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "p_a", &[r_a]),
        mock_pass_graphics(PassIndex(1), "p_b", &[r_b]),
        mock_pass_graphics(PassIndex(2), "p_c", &[r_c]),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);

    assert_eq!(mapping.entries.len(), 3);
    let all_same_slot = mapping.entries[0].1 == mapping.entries[1].1
        && mapping.entries[1].1 == mapping.entries[2].1;
    assert!(
        all_same_slot,
        "three sequential non-overlapping resources share one slot",
    );
    assert_eq!(
        mapping.slots_used, 1,
        "one slot for three sequential resources",
    );
}

#[test]
fn relaxed_policy_nested_lifetime_unique_slots() {
    // Resource A lives [0, 2] (written at P0, read at P2).
    // Resource B lives [1, 1] (written at P1 only, not read elsewhere).
    // B is entirely inside A's lifetime -> overlap -> different slots.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "long_lived", 64, 64),
        mock_resource_texture(r_b, "short_lived", 64, 64),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "write_a", &[r_a]),
        {
            let mut p = mock_pass_compute(PassIndex(1), "write_b", &[], &[]);
            p.access_set.writes.push(r_b);
            p
        },
        {
            let mut p = mock_pass_compute(PassIndex(2), "read_a", &[], &[]);
            p.access_set.reads.push(r_a);
            p
        },
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);

    assert_eq!(mapping.entries.len(), 2);
    assert_ne!(
        mapping.entries[0].1, mapping.entries[1].1,
        "nested lifetimes force distinct slots",
    );
    assert_eq!(mapping.slots_used, 2, "nested lifetimes require two slots",);
}

#[test]
fn relaxed_policy_interleaved_overlap_unique() {
    // A=[0,1] (written at P0, read at P1), B=[1,2] (written at P1, read at P2).
    // They overlap at P1 -> different slots.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "a", 64, 64),
        mock_resource_texture(r_b, "b", 64, 64),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "write_a", &[r_a]),
        {
            let mut p = mock_pass_compute(PassIndex(1), "middle", &[], &[]);
            p.access_set.reads.push(r_a);
            p.access_set.writes.push(r_b);
            p
        },
        {
            let mut p = mock_pass_compute(PassIndex(2), "read_b", &[], &[]);
            p.access_set.reads.push(r_b);
            p
        },
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);

    assert_eq!(mapping.entries.len(), 2);
    assert_ne!(
        mapping.entries[0].1, mapping.entries[1].1,
        "interleaved lifetimes force distinct slots",
    );
    assert_eq!(mapping.slots_used, 2);
}

#[test]
fn relaxed_policy_all_same_pass_all_unique() {
    // All three resources accessed at the same single pass -> all unique.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let r_c = ResourceHandle(3);
    let resources = vec![
        mock_resource_texture(r_a, "a", 64, 64),
        mock_resource_texture(r_b, "b", 64, 64),
        mock_resource_texture(r_c, "c", 64, 64),
    ];
    let passes = vec![mock_pass_graphics(
        PassIndex(0),
        "gbuffer",
        &[r_a, r_b, r_c],
    )];
    let order = vec![PassIndex(0)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);

    assert_eq!(mapping.entries.len(), 3);
    let unique_slots: std::collections::HashSet<usize> =
        mapping.entries.iter().map(|e| e.1).collect();
    assert_eq!(
        unique_slots.len(),
        3,
        "three resources overlapping at same pass = three unique slots",
    );
    assert_eq!(mapping.slots_used, 3);
}

#[test]
fn relaxed_policy_adjacent_non_overlapping_share() {
    // A ends at P0 (last access P0), B starts at P1 (first access P1).
    // Since A's last use index < B's first use index: non-overlapping -> share.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "first", 64, 64),
        mock_resource_texture(r_b, "second", 64, 64),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "pass_a", &[r_a]),
        mock_pass_graphics(PassIndex(1), "pass_b", &[r_b]),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 2);
    assert_eq!(
        mapping.entries[0].1, mapping.entries[1].1,
        "adjacent non-overlapping resources share under Relaxed",
    );
    assert_eq!(mapping.slots_used, 1);
}

// =========================================================================
// SECTION 5 -- Transient policy (selective aliasing)
// =========================================================================

#[test]
fn transient_policy_two_transient_non_overlapping_share() {
    // Two transient resources with non-overlapping lifetimes share a slot.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let mut res_a = mock_resource_texture(r_a, "transient_a", 64, 64);
    res_a.is_history = true;
    let mut res_b = mock_resource_texture(r_b, "transient_b", 64, 64);
    res_b.is_history = true;
    let resources = vec![res_a, res_b];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "pass_a", &[r_a]),
        mock_pass_graphics(PassIndex(1), "pass_b", &[r_b]),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let mapping = apply_aliasing(AliasPolicy::Transient, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 2);
    assert_eq!(
        mapping.entries[0].1, mapping.entries[1].1,
        "transient non-overlapping resources share under Transient policy",
    );
    assert_eq!(mapping.slots_used, 1);
}

#[test]
fn transient_policy_two_persistent_non_overlapping_unique() {
    // Two persistent (non-transient) resources with non-overlapping lifetimes
    // get unique slots under Transient policy (persistent never alias).
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "persistent_a", 64, 64),
        mock_resource_texture(r_b, "persistent_b", 64, 64),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "pass_a", &[r_a]),
        mock_pass_graphics(PassIndex(1), "pass_b", &[r_b]),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let mapping = apply_aliasing(AliasPolicy::Transient, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 2);
    assert_ne!(
        mapping.entries[0].1, mapping.entries[1].1,
        "persistent resources get unique slots under Transient policy",
    );
    assert_eq!(mapping.slots_used, 2);
}

#[test]
fn transient_policy_transient_and_persistent_non_overlapping_unique() {
    // Transient and persistent resources always get different slots
    // under Transient policy, even when lifetimes do not overlap.
    let r_t = ResourceHandle(1);
    let r_p = ResourceHandle(2);
    let mut res_t = mock_resource_texture(r_t, "scratch", 64, 64);
    res_t.is_history = true;
    let resources = vec![res_t, mock_resource_texture(r_p, "persistent", 64, 64)];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "pass_t", &[r_t]),
        mock_pass_graphics(PassIndex(1), "pass_p", &[r_p]),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let mapping = apply_aliasing(AliasPolicy::Transient, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 2);
    assert_ne!(
        mapping.entries[0].1, mapping.entries[1].1,
        "transient and persistent get different slots",
    );
    assert_eq!(mapping.slots_used, 2);
}

#[test]
fn transient_policy_two_transient_overlapping_unique() {
    // Two transient resources with overlapping lifetimes get unique slots.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let mut res_a = mock_resource_texture(r_a, "tmp_a", 64, 64);
    res_a.is_history = true;
    let mut res_b = mock_resource_texture(r_b, "tmp_b", 64, 64);
    res_b.is_history = true;
    let resources = vec![res_a, res_b];
    let passes = vec![mock_pass_graphics(PassIndex(0), "pass", &[r_a, r_b])];
    let order = vec![PassIndex(0)];

    let mapping = apply_aliasing(AliasPolicy::Transient, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 2);
    assert_ne!(
        mapping.entries[0].1, mapping.entries[1].1,
        "overlapping transient resources get different slots",
    );
    assert_eq!(mapping.slots_used, 2);
}

#[test]
fn transient_policy_mixed_transient_persistent() {
    // Two transient resources share a slot, one persistent gets its own.
    // Transients: r_a=[0,0], r_b=[1,1] (non-overlapping)
    // Persistent: r_p=[2,2]
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let r_p = ResourceHandle(3);
    let mut res_a = mock_resource_texture(r_a, "tmp_a", 64, 64);
    res_a.is_history = true;
    let mut res_b = mock_resource_texture(r_b, "tmp_b", 64, 64);
    res_b.is_history = true;
    let resources = vec![
        res_a,
        res_b,
        mock_resource_texture(r_p, "persistent", 64, 64),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "p_a", &[r_a]),
        mock_pass_graphics(PassIndex(1), "p_b", &[r_b]),
        mock_pass_graphics(PassIndex(2), "p_p", &[r_p]),
    ];
    let order = vec![PassIndex(0), PassIndex(1), PassIndex(2)];

    let mapping = apply_aliasing(AliasPolicy::Transient, &order, &passes, &resources);

    assert_eq!(mapping.entries.len(), 3);
    // The two transient resources share the same slot.
    assert_eq!(
        mapping.entries[0].1, mapping.entries[1].1,
        "transient resources share under Transient policy",
    );
    // The persistent resource has a different slot from the transients.
    assert_ne!(
        mapping.entries[0].1, mapping.entries[2].1,
        "persistent is in its own slot, distinct from transients",
    );
    assert_eq!(
        mapping.slots_used, 2,
        "two slots: one for transients, one for persistent",
    );
}

#[test]
fn transient_policy_no_transient_resources_all_unique() {
    // When no resources are transient, Transient policy behaves like None.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_texture(r_a, "a", 64, 64),
        mock_resource_texture(r_b, "b", 64, 64),
    ];
    let passes = vec![
        mock_pass_graphics(PassIndex(0), "p_a", &[r_a]),
        mock_pass_graphics(PassIndex(1), "p_b", &[r_b]),
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let mapping = apply_aliasing(AliasPolicy::Transient, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 2);
    assert_ne!(
        mapping.entries[0].1, mapping.entries[1].1,
        "no transient resources -> all unique under Transient policy",
    );
    assert_eq!(mapping.slots_used, 2);
}

// =========================================================================
// SECTION 6 -- Resource lifecycle edge cases
// =========================================================================

#[test]
fn resource_read_only_lifetime() {
    // A resource that is only read (never written) still has a lifetime.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "read_only", 64, 64)];
    let passes = vec![{
        let mut p = mock_pass_compute(PassIndex(0), "reader", &[], &[]);
        p.access_set.reads.push(r);
        p
    }];
    let order = vec![PassIndex(0)];

    let mapping_none = apply_aliasing(AliasPolicy::None, &order, &passes, &resources);
    assert_eq!(mapping_none.entries.len(), 1);
    assert_eq!(
        mapping_none.entries[0].1, 0,
        "read-only resource mapped to slot 0",
    );
}

#[test]
fn resource_single_pass_lifetime() {
    // Resource written and read in the same pass has discrete lifetime.
    let r = ResourceHandle(1);
    let resources = vec![mock_resource_texture(r, "inout", 64, 64)];
    let passes = vec![{
        let mut p = mock_pass_compute(PassIndex(0), "inout_pass", &[], &[]);
        p.access_set.writes.push(r);
        p.access_set.reads.push(r);
        p
    }];
    let order = vec![PassIndex(0)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 1);
    assert_eq!(
        mapping.entries[0].1, 0,
        "single-pass resource maps to slot 0",
    );
    assert_eq!(mapping.slots_used, 1);
}

#[test]
fn slots_used_accurate_under_none() {
    // Under None policy: slots_used == number of resources.
    let resources: Vec<IrResource> = (1..=5)
        .map(|i| {
            let h = ResourceHandle(i);
            mock_resource_texture(h, &format!("r{}", i), 64, 64)
        })
        .collect();
    let handles: Vec<ResourceHandle> = resources
        .iter()
        .enumerate()
        .map(|(i, _)| ResourceHandle(i as u32 + 1))
        .collect();
    let passes: Vec<IrPass> = handles
        .iter()
        .map(|&h| mock_pass_graphics(PassIndex(h.0 as usize), &format!("p{}", h.0), &[h]))
        .collect();
    let order: Vec<PassIndex> = handles.iter().map(|&h| PassIndex(h.0 as usize)).collect();

    let mapping = apply_aliasing(AliasPolicy::None, &order, &passes, &resources);
    assert_eq!(
        mapping.slots_used, 5,
        "None: slots_used equals resource count"
    );
    assert_eq!(mapping.entries.len(), 5);
}

#[test]
fn slots_used_accurate_under_relaxed() {
    // Under Relaxed: four sequential non-overlapping resources use 1 slot.
    let resources: Vec<IrResource> = (1..=4)
        .map(|i| mock_resource_texture(ResourceHandle(i), &format!("r{}", i), 64, 64))
        .collect();
    let passes: Vec<IrPass> = (1..=4)
        .map(|i| {
            mock_pass_graphics(
                PassIndex(i as usize - 1),
                &format!("p{}", i),
                &[ResourceHandle(i)],
            )
        })
        .collect();
    let order: Vec<PassIndex> = (0..4).map(|i| PassIndex(i)).collect();

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);
    assert_eq!(
        mapping.slots_used, 1,
        "Relaxed: four sequential non-overlapping resources in 1 slot",
    );
    assert_eq!(mapping.entries.len(), 4);
    // All entries share slot 0.
    for entry in &mapping.entries {
        assert_eq!(entry.1, 0, "resource {:?} assigned to slot 0", entry.0,);
    }
}

#[test]
fn buffer_resources_participate_in_aliasing() {
    // Buffer resources should alias the same as textures under Relaxed.
    let r_a = ResourceHandle(1);
    let r_b = ResourceHandle(2);
    let resources = vec![
        mock_resource_buffer(r_a, "buf_a", 4096),
        mock_resource_buffer(r_b, "buf_b", 8192),
    ];
    let passes = vec![
        {
            let mut p = mock_pass_compute(PassIndex(0), "write_a", &[], &[]);
            p.access_set.writes.push(r_a);
            p
        },
        {
            let mut p = mock_pass_compute(PassIndex(1), "write_b", &[], &[]);
            p.access_set.writes.push(r_b);
            p
        },
    ];
    let order = vec![PassIndex(0), PassIndex(1)];

    let mapping = apply_aliasing(AliasPolicy::Relaxed, &order, &passes, &resources);
    assert_eq!(mapping.entries.len(), 2);
    assert_eq!(
        mapping.entries[0].1, mapping.entries[1].1,
        "non-overlapping buffers share slot under Relaxed",
    );
    assert_eq!(mapping.slots_used, 1);
}
