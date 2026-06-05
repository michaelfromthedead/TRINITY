==== SENIOR_QA_SANITY REPORT ====
Task: T-FG-7.7
Role: SENIOR_QA_SANITY
Reviewed findings from: JUNIOR_QA (T-FG-7.7_compilation_result_findings_junior.md)
Files cross-referenced: engine/rendering/framegraph/frame_graph.py
===============================================================================

SANITY FILTER RESULTS
---------------------

Each finding is marked REAL (genuine issue), OVERZEALOUS (not actionable / not a
defect), or DISPUTED (partially valid but severity/classification is wrong).

===============================================================================

HIGH FINDINGS

--- [H2] pass_count fallback silently reports wrong value when cull_stats is missing --
File: engine/rendering/framegraph/frame_graph.py:208
Verdict: REAL (downgrade to MEDIUM)

Analysis:
  pass_count = cull_stats.get("passes_total", len(passes))

  When cull_stats is missing from the bridge response (older bridge version),
  the fallback returns len(passes), which is the count of SURVIVING passes
  only. This silently conflates "no culling happened" with "culling happened
  but we cannot see the pre-cull total" -- the caller sees pass_count equal
  to the surviving count, which is a plausible-looking wrong number.

  The JUNIOR's analysis is technically correct. However, this only triggers
  when the bridge is active but returns data without a cull_stats block,
  which is an incomplete-bridge-response scenario. 0 would be a safer
  default (obviously wrong, signaling "unknown"), but the real fix depends
  on whether the field contract should be "always present when bridge is
  active." Downgraded from HIGH to MEDIUM because:
    - The Python-only fallback path (compile() line 819-823) sets
      pass_count = len(self._passes) correctly.
    - The bug only surfaces in the narrow window of an active-but-incomplete
      bridge response.
    - The value is one of several cull_stats fields; the whole block being
      absent signals a partial response that would likely also affect other
      stats.

--- [H2] alias_group_count is never populated from bridge JSON --
File: engine/rendering/framegraph/frame_graph.py:125 (dataclass field),
     lines 231-240 (from_bridge_json cls() call)
Verdict: REAL (maintain HIGH)

Analysis:
  The dataclass declares alias_group_count with default 0, but
  from_bridge_json never reads it from bridge data or passes it to cls().
  The bridge schema (documented in the from_bridge_json docstring) does not
  list an alias_group_count key, so even a well-formed bridge response
  results in alias_group_count always being 0 on the Python side when the
  bridge is active.

  While the Python-only fallback path (compile() line 835) correctly
  populates this field via
    result.alias_group_count = self._resource_manager.get_alias_group_count()
  the bridge path silently drops to 0. This is a real data-loss bug: if the
  Rust bridge computes aliasing and the result is consumed by a Python caller
  that checks alias_group_count, it gets 0 even though aliasing may have
  occurred.

  The fix is either:
    a) Populate alias_group_count from bridge data if the bridge provides it,
    b) Accept that the bridge path doesn't report this and document the
       limitation in the field's docstring, or
    c) Set alias_group_count = len(passes) // 2 or some heuristic (bad idea).

  Keeping this at HIGH because it's a silent data-loss issue that affects
  every bridge-using caller.

--- [H11] Zero test coverage for T-FG-7.7 code paths --
File: tests/test_framegraph_phase1.py (entire file)
Verdict: REAL (downgrade to MEDIUM)

Analysis:
  The JUNIOR is factually correct: no tests exercise from_bridge_json(),
  CompileError construction, or the new T-FG-7.7 fields. This is a testing
  gap, not a code defect. A testing gap of this nature is standard for a
  DEV commit and is properly the responsibility of a TESTDEV task or
  follow-up issue. It does not block GREEN_LIGHT as long as the acceptance
  criteria are met (72/72 existing tests pass).

  Downgraded from HIGH to MEDIUM because:
    - The code was manually validated by JUNIOR_QA to handle all documented
      input shapes correctly.
    - Zero coverage for new deserialization code is undesirable but expected
      for a DEV-first commit in this pipeline.
    - All existing regression tests pass.
    - This should be filed as a TESTDEV task rather than blocking the commit.

===============================================================================

MEDIUM FINDINGS

--- [M2] Non-dict error entries dropped silently without logging --
File: engine/rendering/framegraph/frame_graph.py:223-224
Verdict: REAL

Analysis:
  The errors loop silently skips entries where isinstance(err_data, dict) is
  False. If the Rust bridge produces a malformed error entry (null, string,
  etc.), the Python side drops it with zero indication. This can hide
  bridge-side bugs during development.

  Using logger.warning for skipped entries is appropriate defensive
  programming. The severity is correctly MEDIUM because:
    - This is a bridge-invariant violation; malformed errors indicate a
      Rust-side bug.
    - Silent drops are observable only by missing errors, which is hard to
      debug.
    - The fix is low-effort (add a logger.warning call).

--- [M2] Non-dict pass entries silently filtered --
File: engine/rendering/framegraph/frame_graph.py:201-204
Verdict: OVERZEALOUS

Analysis:
  The execution_order comprehension filters with isinstance(p, dict) and
  "name" in p. The JUNIOR flags this as the same pattern as the errors
  case, but there is a meaningful difference:

  - Error entries: the bridge produces errors as a data-dependent list;
    individual errors may validly be malformed without breaking the entire
    bridge response. Logging a warning here helps catch real bridge bugs.

  - Pass entries: the passes list is structurally required to contain dicts
    with "name" keys. A pass entry that is not a dict or lacks a name is a
    catastrophic bridge invariant violation -- the compilation result is
    fundamentally broken, not recoverable by logging. Silent filtering is
    acceptable defensive programming for what amounts to garbage-in.
    Adding a warning creates noise for what should be caught by Rust-side
    tests and/or a bridge version check.

  Correctly rated MEDIUM would be if the code actually broke; but the
  filter is correct defensive behavior. OVERZEALOUS.

--- [M2] culled_passes list is empty when bridge is active (data loss) --
File: engine/rendering/framegraph/frame_graph.py:231-240 (cls() call)
Verdict: REAL

Analysis:
  culled_passes (dataclass default field_factory=list) is never set by
  from_bridge_json. When the Rust bridge is active, culled_passes is always
  [], regardless of how many passes were culled on the Rust side.

  The cull count IS reported via culled_count (line 209-211), so callers
  know HOW MANY passes were culled, but the actual NAMES are lost. This is
  data loss for any consumer that needs to know which specific passes were
  eliminated.

  Correctly MEDIUM because:
    - culled_count provides the aggregate, partially mitigating the loss.
    - The bridge schema may not expose culled pass names (a Rust-side gap).
    - The fix requires either bridge schema changes or documentation that
      this field is Python-only.

--- [M5] scope creep: RHIContext type tightening on three methods --
File: engine/rendering/framegraph/frame_graph.py:999, 1027, 1046
Verdict: OVERZEALOUS

Analysis:
  Changing `context: Any` to `context: RHIContext` on execute(),
  _execute_barriers(), and _prepare_for_present() is type tightening, not
  scope creep. Python type annotations are runtime-noop -- there is zero
  API breakage for any caller.

  Arguments for OVERZEALOUS:
    - Type tightening from Any to a concrete type ALWAYS increases type
      safety. This is healthy hygiene, not inappropriate scope expansion.
    - RHIContext is already imported and used internally (line 33). The
      methods were already documented as expecting a rendering context.
    - Any caller passing a non-RHIContext was relying on Any in an
      unprincipled way -- tightening surfaces latent bugs.
    - Creating a separate "typing task" for three parameter annotations
      would be wasteful process overhead.

  The JUNIOR correctly notes the change is outside T-FG-7.7 scope, but
  flagging net-positive inline cleanup as a MEDIUM finding is overzealous.
  This is at most a LOW note.

===============================================================================

LOW FINDINGS

--- [L1] CompileError dataclass has default "" for all three string fields --
File: engine/rendering/framegraph/frame_graph.py:95-101
Verdict: REAL

Analysis:
  Having default "" for all three string fields means CompileError("", "", "")
  is technically valid but meaningless. Making `message` required (no default)
  would enforce that every error carries information.

  Correctly rated LOW. This is a style/ergonomics preference, not a
  functional defect. The defaults are pragmatic for JSON deserialization
  where keys may be absent. If this were to be addressed, the tradeoff
  between deserialization ergonomics and semantic strictness should be
  discussed with the team.

--- [L1] Module-level _PASS_TYPE_TO_STR and _FORMAT_TO_WGPU are IR serialization constants --
File: engine/rendering/framegraph/frame_graph.py:56-80
Verdict: OVERZEALOUS

Analysis:
  These constants support _format_to_wgpu_str() and
  _collect_py_pass_nodes(), which are IR serialization functions that feed
  the Rust bridge -- the same bridge that T-FG-7.7 enriches. They are
  pre-existing infrastructure in the file, not new code introduced by this
  commit (or if newly added, they are a necessary dependency of the bridge
  path, not gratuitous scope expansion).

  Flagging pre-existing or bridge-required constants as "scope creep" is
  overzealous. They belong to the file, and the file is the unit of change.
  This is at most a note for the record, not an actionable finding.

===============================================================================

SUMMARY
-------

Total findings:            9
  REAL (actionable):       5 (1 HIGH, 3 MEDIUM, 1 LOW)
  OVERZEALOUS (dismiss):   3 (1 MEDIUM, 1 MEDIUM, 1 LOW)
  REAL downgraded:         2 (H->M, H->M)

After SANITY filter, actionable findings:

  HIGH (1):
    REAL  - [H2] alias_group_count never populated from bridge JSON      line 125

  MEDIUM (3):
    REAL  - [H] pass_count fallback (downgraded to MEDIUM)               line 208
    REAL  - [M2] Non-dict error entries dropped silently                lines 223-224
    REAL  - [M2] culled_passes empty when bridge active                 lines 231-240

  LOW (1):
    REAL  - [L1] CompileError field defaults                            lines 95-101

DISMISSED as OVERZEALOUS (3):
    - [M2] Non-dict pass entries silently filtered (correct defense)
    - [M5] RHIContext type tightening (net-positive hygiene)
    - [L1] IR constants scope creep (bridge-required infrastructure)

OUTSTANDING ITEMS FOR SENIOR_QA_FINAL
--------------------------------------

1. The alias_group_count HIGH finding is the only remaining HIGH after
   SANITY filtering. The fix is low-effort (one line in from_bridge_json),
   but the bridge schema may need a corresponding Rust-side key.

2. The pass_count fallback (downgraded to MEDIUM) and culled_passes
   (MEDIUM) are both bridge-schema limitations -- the Python side cannot
   populate them correctly without corresponding Rust-side data. These
   may need to be documented field limitations rather than code fixes.

3. The non-dict error warning (MEDIUM) is a simple add-a-log-line fix
   with no bridge dependency.

4. Zero test coverage (downgraded from HIGH to MEDIUM) should be filed
   as a separate TESTDEV task rather than blocking GREEN_LIGHT -- all
   72 existing tests pass and manual validation confirms the logic is
   sound.
