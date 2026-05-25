# Concerning Progress — Systemic Issues Discovered 2026-05-22

## 1. False TODO Toggles — Checkboxes Lying About Completion

The checkmark-fix worker toggled `[x]` marks by scanning for GREEN_LIGHT commits repo-wide
but did NOT verify those commits were reachable from `main`. This caused false toggles:

- **T-DEMO-1.2 (sdf_box)**: Toggled `[x]` but the WGSL file and tests were on a
  task branch never merged to main. Code existed but was not delivered. Now fixed.
- **Potentially others**: The worker toggled 11 tasks across 5 gap sets. Only a
  subset have been manually verified. The remainder may also be false.

**Root cause**: `git log --all --grep="GREEN_LIGHT"` finds commits on any branch,
not just those merged to main. Verification must use `git merge-base --is-ancestor`
or `git log main --grep="GREEN_LIGHT"` to confirm the commit is reachable.

## 2. Code Lost During Squash Merge — Branch Deleted, Code Missing

**T-DEMO-2.4 (combinator codegen)** went through the full DEV→TEST→JUNIOR→SANITY→FINAL
pipeline and received GREEN_LIGHT. 302/306 combinator tests passed. The task branch
was deleted per protocol. But the `git merge --squash` silently failed to include
the codegen file changes. The `min2`/`max2`/`smin`/`smax` WGSL functions do not
exist anywhere in git — not on main, not on any branch, not in any dangling commit.
The code is genuinely lost and needs full re-implementation.

**Root cause**: `git merge --squash task/T-DEMO-2.4` produced "nothing to squash"
or conflicted, but the error was not caught. The merge commit was created anyway
with only partial file inclusion. The branch was then deleted. The code existed
only on that branch and is now unrecoverable from git history.

## 3. Squash Merge Reliability

Squash merges silently tolerate several failure modes:

- **"Already up to date"**: Files from the task branch are already on main from
  a previous squash merge of a sibling branch. No error, but the commit may be
  empty or missing new changes.
- **Merge conflicts**: When files exist on both sides with conflicting content,
  `git merge --squash` drops into conflict state. The automated scripts did not
  detect this and created empty commits.
- **Partial file inclusion**: When a squash merge resolves some files but not
  others (due to conflicts or pre-existing content), the missing files are
  silently dropped.

**Required fix**: After every squash merge, verify that the key deliverable files
from the task branch exist on main using:
```
git show main:<path/to/file> 2>/dev/null || echo "MISSING"
```

## 4. Branch Deletion Before Verification

The GREEN_LIGHT protocol deletes the task branch immediately after squash merge.
If the merge was incomplete, the branch (and its code) are lost. The protocol
must verify file presence on main BEFORE deleting the task branch.

## 5. Checkmark Worker Trusted Commit Existence Over Merge Status

The worker found GREEN_LIGHT commit messages in the repo and assumed any commit
with that message meant the code was delivered. It did not check:
- Whether the commit was reachable from main
- Whether the deliverable files actually exist on main
- Whether the toggle was already correct before toggling

## 6. Working Tree vs. Main Divergence

TODO files read from the working tree reflect the currently-checked-out branch,
not main. Scans from task branches show stale TODO states. This caused confusion
about actual completion percentages throughout the session.

**Required fix**: Always read TODO files from main:
```
git show main:docs/gap_sets/<GAPSET>/PHASE_N_TODO.md
```

## Current Verified State (2026-05-22)

- 15 GREEN_LIGHT tasks verified (code confirmed on main via file existence checks)
- 1 task lost (T-DEMO-2.4 combinator codegen)
- 1 task recovered from false toggle (T-DEMO-1.2 sdf_box)
- Audit of remaining [x] toggles incomplete — potential for more false positives
- 6 gap sets in table format (FRAME_GRAPH, LIGHTING, TOOLING, etc.) not yet
  converted to checkbox format — their task counts are excluded from percentages
- True completion: ~5% without BRIDGE (which was bulk-restored and unverified)

## Actions Needed

1. Re-implement T-DEMO-2.4 combinator codegen (min2/max2/smin/smax WGSL functions)
2. Audit every [x] toggle against `git show main:<deliverable>` verification
3. Fix squash merge protocol to verify file presence before branch deletion
4. Convert remaining 6 table-format gap sets to checkbox format
5. Only trust TODO counts read from main, never from task branches
6. Add merge-base ancestry check to any automated toggle process
