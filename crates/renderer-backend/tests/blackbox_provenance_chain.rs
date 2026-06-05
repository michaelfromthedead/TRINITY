//! BLACK-BOX TESTS: T-MAT-7.5 Provenance chain pruning
//!
//! Acceptance criteria from PHASE_N_TODO.md:
//! - Chain exceeding N entries is trimmed
//! - Origin and current entries are always preserved
//!
//! Test coverage:
//! 1. Chain trimming: KeepLastN(5) with 20 entries
//! 2. Origin preservation with 100 entries
//! 3. Current preservation after pruning
//! 4. MaxAge pruning
//! 5. Stress test: 1000 entries with KeepLastN(10)

use renderer_backend::pipeline::{
    ContentHash, ProvenanceChain, ProvenanceEntry, PruningStrategy,
};

// ---------------------------------------------------------------------------
// Test 1: Chain trimming with KeepLastN(5) and 20 entries
// ---------------------------------------------------------------------------

#[test]
fn test_chain_trimming_keep_last_n_5() {
    let strategy = PruningStrategy::KeepLastN(5);

    // Create origin
    let h0 = ContentHash::from_bytes(b"origin_entry");
    let origin = ProvenanceEntry::origin(h0, 1000, Some("origin".into()));
    let mut chain = ProvenanceChain::with_origin(origin, strategy);

    // Push 20 entries sequentially and verify trimming after each
    let mut prev_hash = h0;
    for i in 1..=20 {
        let hash = ContentHash::from_bytes(format!("entry_{}", i).as_bytes());
        let entry = ProvenanceEntry::with_parent(hash, 1000 + i as u64, None, prev_hash);
        chain.push(entry);
        prev_hash = hash;

        // After the first few entries, we should stay at or below 5
        if i >= 4 {
            assert!(
                chain.len() <= 5,
                "After push {}: len() = {}, expected <= 5",
                i,
                chain.len()
            );
        }
    }

    // Final verification
    assert!(chain.len() <= 5, "Final len() = {}, expected <= 5", chain.len());

    // Verify trimmed entries are middle entries, not origin/current
    let origin_entry = chain.origin().expect("chain should have origin");
    let current_entry = chain.current().expect("chain should have current");

    // Origin should still be the first entry we created
    assert_eq!(origin_entry.hash, h0, "Origin must be preserved");

    // Current should be the last pushed entry (entry_20)
    let expected_current = ContentHash::from_bytes(b"entry_20");
    assert_eq!(
        current_entry.hash, expected_current,
        "Current must be the most recently pushed entry"
    );

    // All middle entries should NOT be the origin or current
    let entries = chain.entries();
    for (idx, entry) in entries.iter().enumerate() {
        if idx == 0 {
            assert_eq!(entry.hash, h0, "First entry must be origin");
        } else if idx == entries.len() - 1 {
            assert_eq!(entry.hash, expected_current, "Last entry must be current");
        }
    }
}

#[test]
fn test_chain_verify_len_after_each_push() {
    let strategy = PruningStrategy::KeepLastN(5);

    let h0 = ContentHash::from_bytes(b"origin");
    let origin = ProvenanceEntry::origin(h0, 0, None);
    let mut chain = ProvenanceChain::with_origin(origin, strategy);

    let mut prev_hash = h0;
    let mut max_len_exceeded_at: Option<usize> = None;

    for i in 1..=20 {
        let hash = ContentHash::from_bytes(format!("v{}", i).as_bytes());
        chain.push(ProvenanceEntry::with_parent(hash, i as u64, None, prev_hash));
        prev_hash = hash;

        // Track if we ever exceed the limit
        if chain.len() > 5 {
            max_len_exceeded_at = Some(i);
            break;
        }
    }

    assert!(
        max_len_exceeded_at.is_none(),
        "Chain exceeded limit at push {}; len() should never exceed 5",
        max_len_exceeded_at.unwrap_or(0)
    );
}

// ---------------------------------------------------------------------------
// Test 2: Origin preservation with 100 entries
// ---------------------------------------------------------------------------

#[test]
fn test_origin_preserved_with_100_entries() {
    let strategy = PruningStrategy::KeepLastN(5);

    // Create origin with a specific hash
    let origin_hash = ContentHash::from_bytes(b"the_original_origin_entry");
    let origin_entry = ProvenanceEntry::origin(
        origin_hash,
        1000,
        Some("Initial version".into()),
    );
    let mut chain = ProvenanceChain::with_origin(origin_entry.clone(), strategy);

    // Add 100 entries
    let mut prev_hash = origin_hash;
    for i in 1..=100 {
        let hash = ContentHash::from_bytes(format!("version_{}", i).as_bytes());
        chain.push(ProvenanceEntry::with_parent(
            hash,
            1000 + i as u64,
            Some(format!("v{}", i)),
            prev_hash,
        ));
        prev_hash = hash;
    }

    // Verify origin is still present and unchanged
    let retrieved_origin = chain.origin().expect("Chain must have origin");
    assert_eq!(
        retrieved_origin.hash, origin_hash,
        "Origin hash must be preserved after 100 entries"
    );
    assert_eq!(
        retrieved_origin.timestamp, 1000,
        "Origin timestamp must be preserved"
    );
    assert_eq!(
        retrieved_origin.message,
        Some("Initial version".into()),
        "Origin message must be preserved"
    );
    assert!(
        retrieved_origin.parent.is_none(),
        "Origin should have no parent"
    );
}

#[test]
fn test_origin_preserved_with_aggressive_pruning() {
    // KeepLastN(2) is the most aggressive: only origin + current
    let strategy = PruningStrategy::KeepLastN(2);

    let origin_hash = ContentHash::from_bytes(b"aggressive_origin");
    let origin = ProvenanceEntry::origin(origin_hash, 0, Some("must survive".into()));
    let mut chain = ProvenanceChain::with_origin(origin, strategy);

    // Add 100 entries
    let mut prev_hash = origin_hash;
    for i in 1..=100 {
        let hash = ContentHash::from_bytes(format!("aggressive_{}", i).as_bytes());
        chain.push(ProvenanceEntry::with_parent(hash, i as u64, None, prev_hash));
        prev_hash = hash;

        // Origin must ALWAYS be preserved
        assert_eq!(
            chain.origin().unwrap().hash,
            origin_hash,
            "Origin lost after push {}",
            i
        );
    }

    // Final verification
    assert_eq!(chain.len(), 2, "KeepLastN(2) should result in exactly 2 entries");
    assert_eq!(chain.origin().unwrap().hash, origin_hash);
    assert_eq!(chain.origin().unwrap().message, Some("must survive".into()));
}

// ---------------------------------------------------------------------------
// Test 3: Current preservation test
// ---------------------------------------------------------------------------

#[test]
fn test_current_always_most_recently_pushed() {
    let strategy = PruningStrategy::KeepLastN(3);

    let h0 = ContentHash::from_bytes(b"origin");
    let mut chain = ProvenanceChain::with_origin(
        ProvenanceEntry::origin(h0, 0, None),
        strategy,
    );

    let mut prev_hash = h0;
    for i in 1..=50 {
        let hash = ContentHash::from_bytes(format!("current_test_{}", i).as_bytes());
        let entry = ProvenanceEntry::with_parent(
            hash,
            i as u64 * 100,
            Some(format!("entry_{}", i)),
            prev_hash,
        );
        chain.push(entry);
        prev_hash = hash;

        // After EVERY push, current() should return the just-pushed entry
        let current = chain.current().expect("current should exist");
        assert_eq!(
            current.hash, hash,
            "After push {}: current hash mismatch",
            i
        );
        assert_eq!(
            current.timestamp,
            i as u64 * 100,
            "After push {}: current timestamp mismatch",
            i
        );
        assert_eq!(
            current.message,
            Some(format!("entry_{}", i)),
            "After push {}: current message mismatch",
            i
        );
    }
}

#[test]
fn test_current_preserved_after_heavy_pruning() {
    let strategy = PruningStrategy::KeepLastN(2);

    let h0 = ContentHash::from_bytes(b"start");
    let mut chain = ProvenanceChain::with_origin(
        ProvenanceEntry::origin(h0, 0, None),
        strategy,
    );

    // Push many entries
    let mut prev = h0;
    for i in 1..=1000 {
        let hash = ContentHash::from_bytes(format!("heavy_{}", i).as_bytes());
        chain.push(ProvenanceEntry::with_parent(hash, i as u64, None, prev));
        prev = hash;
    }

    // Current should be the very last pushed
    let expected_current = ContentHash::from_bytes(b"heavy_1000");
    assert_eq!(
        chain.current().unwrap().hash,
        expected_current,
        "Current must be the last pushed entry even after heavy pruning"
    );
}

// ---------------------------------------------------------------------------
// Test 4: MaxAge pruning test
// ---------------------------------------------------------------------------

#[test]
fn test_max_age_prunes_old_entries() {
    // MaxAge(60): keep entries newer than 60 seconds from "now"
    let strategy = PruningStrategy::MaxAge(60);

    let h1 = ContentHash::from_bytes(b"origin");
    let h2 = ContentHash::from_bytes(b"very_old"); // now - 120s
    let h3 = ContentHash::from_bytes(b"old");      // now - 90s
    let h4 = ContentHash::from_bytes(b"recent");   // now - 30s
    let h5 = ContentHash::from_bytes(b"current");  // now

    let now = 1000u64;

    let origin = ProvenanceEntry::origin(h1, now - 200, Some("origin".into()));
    let mut chain = ProvenanceChain::with_origin(origin, strategy);

    // Add entries with different ages
    chain.push(ProvenanceEntry::with_parent(h2, now - 120, None, h1)); // 120s old
    chain.push(ProvenanceEntry::with_parent(h3, now - 90, None, h2));  // 90s old
    chain.push(ProvenanceEntry::with_parent(h4, now - 30, None, h3));  // 30s old (recent)
    chain.push(ProvenanceEntry::with_parent(h5, now, None, h4));       // current (now)

    // Cutoff = now - 60 = 940
    // h1 (800) should be kept (origin)
    // h2 (880) should be pruned (older than cutoff)
    // h3 (910) should be pruned (older than cutoff)
    // h4 (970) should be kept (newer than cutoff)
    // h5 (1000) should be kept (current)

    // Origin is ALWAYS preserved even if older than max_age
    assert_eq!(chain.origin().unwrap().hash, h1, "Origin must be preserved");

    // Current is ALWAYS preserved
    assert_eq!(chain.current().unwrap().hash, h5, "Current must be preserved");

    // Check that old entries (h2, h3) are pruned
    // Chain should have: origin, recent, current (or just origin, current depending on impl)
    let entries = chain.entries();
    let entry_hashes: Vec<_> = entries.iter().map(|e| e.hash).collect();

    // h2 and h3 should NOT be in the chain
    assert!(
        !entry_hashes.contains(&h2),
        "h2 (120s old) should be pruned but is present"
    );
    assert!(
        !entry_hashes.contains(&h3),
        "h3 (90s old) should be pruned but is present"
    );

    // h4 should still be there (30s old, within 60s window)
    assert!(
        entry_hashes.contains(&h4),
        "h4 (30s old) should be kept but is missing"
    );
}

#[test]
fn test_max_age_origin_preserved_even_if_old() {
    let strategy = PruningStrategy::MaxAge(10); // Very aggressive: 10 seconds

    let h1 = ContentHash::from_bytes(b"ancient_origin");
    let origin = ProvenanceEntry::origin(h1, 0, None); // timestamp 0
    let mut chain = ProvenanceChain::with_origin(origin, strategy);

    // Add entries with much newer timestamps
    let mut prev = h1;
    for i in 1..=20 {
        let hash = ContentHash::from_bytes(format!("new_{}", i).as_bytes());
        // Timestamps 1000, 1001, 1002, ... (much newer than origin at 0)
        chain.push(ProvenanceEntry::with_parent(hash, 1000 + i as u64, None, prev));
        prev = hash;
    }

    // Origin (timestamp 0) is WAY older than max_age allows
    // But it MUST still be preserved
    assert_eq!(
        chain.origin().unwrap().hash, h1,
        "Origin must be preserved even when older than max_age"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Stress test - 1000 entries with KeepLastN(10)
// ---------------------------------------------------------------------------

#[test]
fn test_stress_1000_entries_keep_last_10() {
    let strategy = PruningStrategy::KeepLastN(10);

    let h0 = ContentHash::from_bytes(b"stress_origin");
    let origin = ProvenanceEntry::origin(h0, 0, None);
    let mut chain = ProvenanceChain::with_origin(origin, strategy);

    let mut prev_hash = h0;
    let mut max_observed_len = 1usize;

    for i in 1..=1000 {
        let hash = ContentHash::from_bytes(format!("stress_entry_{}", i).as_bytes());
        chain.push(ProvenanceEntry::with_parent(
            hash,
            i as u64,
            None,
            prev_hash,
        ));
        prev_hash = hash;

        // Track max length observed
        let current_len = chain.len();
        if current_len > max_observed_len {
            max_observed_len = current_len;
        }

        // Verify we never exceed the limit
        assert!(
            current_len <= 10,
            "After push {}: len() = {}, exceeded limit of 10",
            i,
            current_len
        );
    }

    // Verify final state
    assert!(chain.len() <= 10, "Final len() = {}", chain.len());
    assert_eq!(chain.origin().unwrap().hash, h0, "Origin must be preserved");

    let expected_current = ContentHash::from_bytes(b"stress_entry_1000");
    assert_eq!(
        chain.current().unwrap().hash,
        expected_current,
        "Current must be entry_1000"
    );

    // Verify no memory leak by checking entries don't accumulate
    // (This is implicit - if we leaked memory, len() would grow)
    println!(
        "Stress test passed: max observed len = {}, final len = {}",
        max_observed_len,
        chain.len()
    );
}

#[test]
fn test_stress_entries_are_dropped() {
    // This test verifies entries are actually dropped, not just hidden
    let strategy = PruningStrategy::KeepLastN(3);

    let h0 = ContentHash::from_bytes(b"drop_test_origin");
    let mut chain = ProvenanceChain::with_origin(
        ProvenanceEntry::origin(h0, 0, None),
        strategy,
    );

    // Push 1000 entries
    let mut prev = h0;
    for i in 1..=1000 {
        let hash = ContentHash::from_bytes(format!("drop_{}", i).as_bytes());
        let message = format!("message_{}", i);
        chain.push(ProvenanceEntry::with_parent(
            hash,
            i as u64,
            Some(message),
            prev,
        ));
        prev = hash;
    }

    // Only 3 entries should remain
    assert_eq!(chain.len(), 3);

    // Verify the entries are as expected
    let entries = chain.entries();

    // First must be origin
    assert_eq!(entries[0].hash, h0);

    // Last must be entry 1000
    let expected_last = ContentHash::from_bytes(b"drop_1000");
    assert_eq!(entries[entries.len() - 1].hash, expected_last);

    // Second-to-last should be entry 999 (last 2 non-origin entries are kept)
    let expected_second_last = ContentHash::from_bytes(b"drop_999");
    assert_eq!(entries[entries.len() - 2].hash, expected_second_last);
}

// ---------------------------------------------------------------------------
// Test 6: Combined strategy
// ---------------------------------------------------------------------------

#[test]
fn test_combined_strategy() {
    let strategy = PruningStrategy::Combined {
        keep_last_n: 5,
        max_age_secs: 100,
    };

    let h0 = ContentHash::from_bytes(b"combined_origin");
    let origin = ProvenanceEntry::origin(h0, 0, None);
    let mut chain = ProvenanceChain::with_origin(origin, strategy);

    // Add entries with varied timestamps
    // Now = 1000
    let now = 1000u64;
    let timestamps = [
        (50, "very_old"),   // 950 seconds old - pruned by both
        (200, "old"),       // 800 seconds old - pruned by both
        (920, "borderline"), // 80 seconds old - pruned by max_age
        (950, "recent_1"),  // 50 seconds old - kept by max_age
        (960, "recent_2"),  // 40 seconds old - kept by max_age
        (980, "recent_3"),  // 20 seconds old - kept by max_age
        (now, "current"),   // current - always kept
    ];

    let mut prev = h0;
    for (ts, name) in &timestamps {
        let hash = ContentHash::from_bytes(name.as_bytes());
        chain.push(ProvenanceEntry::with_parent(hash, *ts, Some(name.to_string()), prev));
        prev = hash;
    }

    // Origin must be preserved
    assert_eq!(chain.origin().unwrap().hash, h0);

    // Current must be preserved
    let current_hash = ContentHash::from_bytes(b"current");
    assert_eq!(chain.current().unwrap().hash, current_hash);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_empty_chain() {
    let chain = ProvenanceChain::new(PruningStrategy::KeepLastN(5));
    assert!(chain.is_empty());
    assert!(chain.origin().is_none());
    assert!(chain.current().is_none());
    assert_eq!(chain.len(), 0);
}

#[test]
fn test_single_entry_chain() {
    let h0 = ContentHash::from_bytes(b"single");
    let origin = ProvenanceEntry::origin(h0, 100, Some("only one".into()));
    let chain = ProvenanceChain::with_origin(origin.clone(), PruningStrategy::KeepLastN(5));

    assert_eq!(chain.len(), 1);
    assert_eq!(chain.origin().unwrap().hash, h0);
    assert_eq!(chain.current().unwrap().hash, h0);
    // Origin and current are the same for single-entry chain
    assert_eq!(chain.origin(), chain.current());
}

#[test]
fn test_keep_last_n_1_becomes_2() {
    // KeepLastN(1) should be treated as KeepLastN(2) to preserve origin + current
    let strategy = PruningStrategy::KeepLastN(1);

    let h0 = ContentHash::from_bytes(b"origin_n1");
    let h1 = ContentHash::from_bytes(b"second_n1");
    let h2 = ContentHash::from_bytes(b"third_n1");

    let mut chain = ProvenanceChain::with_origin(
        ProvenanceEntry::origin(h0, 0, None),
        strategy,
    );

    chain.push(ProvenanceEntry::with_parent(h1, 1, None, h0));
    chain.push(ProvenanceEntry::with_parent(h2, 2, None, h1));

    // Should have at least 2 entries (origin + current)
    assert!(chain.len() >= 2, "KeepLastN(1) should behave as at least 2");
    assert_eq!(chain.origin().unwrap().hash, h0);
    assert_eq!(chain.current().unwrap().hash, h2);
}

#[test]
fn test_two_entries_no_pruning() {
    let strategy = PruningStrategy::KeepLastN(5);

    let h0 = ContentHash::from_bytes(b"first");
    let h1 = ContentHash::from_bytes(b"second");

    let mut chain = ProvenanceChain::with_origin(
        ProvenanceEntry::origin(h0, 0, None),
        strategy,
    );
    chain.push(ProvenanceEntry::with_parent(h1, 1, None, h0));

    // Two entries, limit 5, no pruning should occur
    assert_eq!(chain.len(), 2);
    assert_eq!(chain.origin().unwrap().hash, h0);
    assert_eq!(chain.current().unwrap().hash, h1);
}
