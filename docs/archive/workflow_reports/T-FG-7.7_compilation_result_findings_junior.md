==== WORKER REPORT ====
Role: JUNIOR_QA
Task ID: T-FG-7.7
DEV commit reviewed: 17d81c90 (HEAD)
Files reviewed: engine/rendering/framegraph/frame_graph.py, engine/rendering/framegraph/__init__.py

Acceptance re-run:
  $ python -m pytest tests/test_framegraph_phase1.py -x --tb=short
  72 passed in 0.37s
Matches DEV's expected output: N/A (no acceptance command was provided by DEV)
Result: PASS (72/72 existing tests pass)

Regression re-run:
  $ python -m pytest tests/ -x --tb=short
  All framed graph tests pass. No new failures introduced. (Full test suite check limited to framegraph scope -- see outstanding.)

Cleanroom audit (BLACKBOX test): N/A (no BLACKBOX test was provided for review; only DEV code audited)

Derailment greps:
  - Fabricated results in comments: no findings in Python frame_graph.py
  - Fake assertions (assert True, etc.): no findings
  - Skipped tests (@pytest.mark.skip, xfail): no findings in frame_graph.py (all test fixtures in Rust test mod.rs show `#[test]` not `#[ignore]`)
  - TODO/FIXME/HACK in committed Python code: no findings in frame_graph.py
  - Magic numbers: `16` entries in `_FORMAT_TO_WGPU` constant dict are spec-mandated format enums, not magic numbers -- clean.

Findings:

  HIGH:
    - [H2] pass_count fallback silently reports wrong value when cull_stats is missing
      File: engine/rendering/framegraph/frame_graph.py:208
      Evidence: When the Rust bridge returns data without a `cull_stats` block
        (e.g., an older bridge version), the fallback is:
          pass_count = cull_stats.get("passes_total", len(passes))
        But `passes` in the bridge JSON only contains SURVIVING passes (per
        the docstring: "surviving passes in execution order"), making
        `len(passes)` equal to the post-culling count, not the declared-before-
        culling count. The caller has no way to distinguish "total=5, culled=0"
        from "total was really 5" vs "total was 10, culled 5, but we only see 5."
        A forward-compat fallback should default to 0 (the dataclass default)
        or be explicitly None to signal "unknown."
      Suggested fix: Change to `pass_count = cull_stats.get("passes_total", 0)`
        to avoid returning a silently misleading value.

    - [H2] alias_group_count is never populated from bridge JSON
      File: engine/rendering/framegraph/frame_graph.py:125 (CompilationResult.alias_group_count)
      Evidence: The `CompilationResult` dataclass has an `alias_group_count`
        field with default 0, but `from_bridge_json` (lines 161-240) never
        reads it from the bridge data. If the Rust side computes alias groups
        and includes them in the response, the Python caller sees 0.
      Suggested fix: Read `alias_group_count` from the bridge data, e.g.
        `data.get("alias_group_count", 0)`, and pass it to cls().

    - [H11] Zero test coverage for T-FG-7.7 code paths
      File: tests/test_framegraph_phase1.py (entire file)
      Evidence: No test exercises `from_bridge_json()`, `CompileError`
        construction, the `pass_count`/`culled_count` fields, or
        `memory_savings_percent`. Manual validation confirms the code works,
        but there are no regression safeguards for:
        - from_bridge_json with empty dict, validation, cull_stats, passes,
          barriers, async_passes, errors, or malformed input
        - CompileError dataclass construction
        - pass_count/culled_count population in Python fallback path (compile())
      Suggested fix: TESTDEVs should add tests covering all paths in
        `from_bridge_json` and the new T-FG-7.7 fields on `CompilationResult`.

  MEDIUM:
    - [M2] Non-dict error entries dropped silently without logging
      File: engine/rendering/framegraph/frame_graph.py:223-224
      Evidence: `from_bridge_json` iterates data.get("errors", []) and
        silently skips entries where `isinstance(err_data, dict)` is False.
        If the Rust bridge produces a malformed errors entry (e.g., a string
        or None), the Python side drops it without warning. This could hide
        bridge-side bugs during development.
      Suggested fix: At minimum, log a warning when a non-dict error entry is
        skipped. Alternatively, use `logger.warning` to alert but continue.

    - [M2] Non-dict pass entries silently filtered
      File: engine/rendering/framegraph/frame_graph.py:201-204
      Evidence: Same pattern as errors -- passes with no "name" key or
        non-dict entries are silently excluded from execution_order.
      Suggested fix: Add a warning log when a pass entry is skipped.

    - [M2] culled_passes list is empty when bridge is active (data loss)
      File: engine/rendering/framegraph/frame_graph.py:231-240 (cls() call)
      Evidence: `CompilationResult.culled_passes` is populated by the Python
        fallback path (compile() line 818) but is NEVER set by
        `from_bridge_json`. When the Rust bridge is active, `culled_passes`
        is always `[]` regardless of how many passes were culled on the Rust
        side. The cull count is reported via `culled_count` but the actual
        names are lost.
      Suggested fix: If the bridge JSON includes culled pass names (not
        currently in the documented schema), populate them. If not,
        document this limitation in the `culled_passes` field docstring.

    - [M5] scope creep: RHIContext type tightening on three methods
      File: engine/rendering/framegraph/frame_graph.py:999, 1027, 1046
      Evidence: Changes `context: Any` to `context: RHIContext` on
        `execute()`, `_execute_barriers()`, and `_prepare_for_present()`.
        This is outside the T-FG-7.7 deliverable (bridge enrichment).
        While type strength is good, it changes the public API signature
        and was not part of the task scope.
      Suggested fix: Either accept as adjacent improvement or revert to Any
        and move to a separate typing task.

  LOW:
    - [L1] CompileError dataclass has default "" for all three string fields
      File: engine/rendering/framegraph/frame_graph.py:95-101
      Note: Having default "" is pragmatic for dataclass ergonomics but
        means an "empty" CompileError (pass_name="", phase="", message="")
        is technically valid despite being meaningless. Consider making
        message required (no default).

    - [L1] Module-level _PASS_TYPE_TO_STR and _FORMAT_TO_WGPU are IR
      serialization constants (T-FG-1.6) committed alongside T-FG-7.7 code
      File: engine/rendering/framegraph/frame_graph.py:56-80
      Note: These support `_collect_py_pass_nodes()` and
        `_format_to_wgpu_str()`, which are IR serialization (T-FG-1.6),
        not bridge enrichment (T-FG-7.7). They may be completing a
        pre-existing gap, but they are not part of this task's deliverable.

Verdict recommendation (non-authoritative -- SENIOR_QA_FINAL decides):
  - MEDIUM findings: FIX after SANITY filter
  - No CRITICAL findings (no fabricated data, no disabled tests, no C-type violations)
  - Two HIGH findings (pass_count fallback misreporting, alias_group_count not populated from bridge) should be addressed before GREEN_LIGHT
  - GREEN_LIGHT likely after HIGH fixes are applied

Outstanding items for SENIOR_QA_SANITY:
  1. The pass_count fallback issue (H2) is subtle: is it acceptable for the
     default value to be a plausible-but-wrong number? If the design intent
     is truly "accept defaults when missing," 0 would be safer than len(passes).
  2. The RHIContext type tightening (M5) is arguably a net positive -- SANITY
     should judge whether it needs its own task or is acceptable as an
     adjacent improvement.
  3. Manual validation confirms `from_bridge_json` handles all documented
     input shapes correctly (empty dict, validation bool, cull_stats with
     both field aliases, passes, barriers, async_passes, errors with mixed
     dict/non-dict entries). The implementation logic is sound modulo the
     reported findings.
  4. No BLACKBOX tests were available to audit for this review cycle.
