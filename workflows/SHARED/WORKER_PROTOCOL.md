# Swarm Worker — Shared Protocol

**Purpose:** The contract every worker follows, regardless of role. Non-negotiable baseline.

**Who reads this:** every DEV, TESTDEV_{WHITEBOX,BLACKBOX}, JUNIOR_QA, SENIOR_QA_{SANITY,FINAL}, ARCH_DEV, and QUEEN — before doing anything.

**Read after:** `workflows/SHARED/WORKER.md` (master index), `workflows/SDLC/SDLC_WORKFLOW.json` (authoritative workflow).
**Read before:** your role-specific doc.

---

## 1. Hard rules — non-negotiable

Violations mean: stop, report, do not continue.

### 1.1 Honest reporting

- **Every number in your report comes from a command you actually ran.** If your report says `p50 = 3.2μs`, you ran the bench and saw `3.2`. No estimates. No "should be around…"
- **Quote the output verbatim** in a fenced code block. QUEEN and downstream workers re-run and compare.
- **If you couldn't run it, say so explicitly.** "Could not run X because Y" is a valid report. "X passes" when you didn't run X is a lie.

### 1.2 Code hygiene

- **No `TODO`, `FIXME`, `HACK`, `WONTFIX`, `XXX` in committed code.** Unfinished work goes in the TODO doc, not the source.
- **No comments describing uncomputed results.** `// result: rank = 500` is only valid if you ran it and saw 500.
- **No placeholder / stub code surviving past its creation task.** If you write a stub, it ships with an `xfail` test or visible gate. No `ssm_decode.py`-style "simplified for now" — reference `EVALUATION.md` for why.
- **No magic numbers where a config value exists.** Use `QwenConfig.hidden`, not `2048`.
- **No disabled tests** via `@pytest.mark.skip`, empty bodies, `assert True`. If it doesn't pass, fix the code; don't silence the test.
- **No fake test data.** Asserted values must be what the code actually produces.

### 1.3 Scope

- **One task at a time.** Your TASK_ID is in the prompt.
- **The task's Do NOT list is authoritative.** Even if touching forbidden territory would obviously help — don't.
- **Scope creep is derailment**, not helpfulness. It poisons downstream QA and FIX cycles.
- **If you see something needing fixing but not in your task:** note it in your report's "outstanding" section. Don't silently do it.

### 1.4 Validation

- **The harness is the primary validation surface.** See `PHASE_1_STRIP_ROCM_ARCH.md` §0 and `PHASE_2_DEEPSEEKSTYLE_ARCH.md` §0. Extend the harness before writing one-off scripts.
- **The task's acceptance criterion is the definition of done.** Not "I think it works." The command listed must produce the expected output.
- **Run the acceptance command; paste the output.**

### 1.5 Blocker handling

- **Stop and report.** Don't pretend to finish. Don't leave the branch broken without saying so.
- **Describe the blocker fully.** Command that failed, error message, what you tried, what you concluded.
- **Preserve your work.** Commit WIP with `[BLOCKED]` prefix and describe unfinished parts.

### 1.6 Estimates

- **Each task has optimistic/realistic/pessimistic estimates.**
- **Blown pessimistic → stop.** The task is wrong — missing prereq, larger scope than written, hidden blocker. Raise it, don't grind.

### 1.7 QA-specific rules (for QA roles)

- **JUNIOR_QA:** high recall; over-flagging is by design. SANITY filters later.
- **SENIOR_QA_SANITY:** judge each junior finding; don't add new findings.
- **SENIOR_QA_FINAL:** emit exactly one verdict — `GREEN_LIGHT | FIX | REWRITE | ESCALATE`. Verdict goes in the report.

### 1.8 Cleanroom rules (for TESTDEV_BLACKBOX)

- **Forbidden files listed in prompt must not be read.** Even a quick peek poisons the blackbox discipline.
- **QA audits for visibility leaks.** Tests that encode internal-specific shapes are tells.

---

## 2. Tools

### 2.1 File tools

- `Read` for any file. Absolute paths.
- `Edit` for modifying existing files. Small, targeted edits. Preserve formatting.
- `Write` only for new files or complete rewrites. Prefer `Edit`.
- `Grep` / `Glob` for search — not `bash grep` / `find`.

### 2.2 Bash

- `Bash` tool for commands. Not for file ops (use dedicated tools).
- Long-running commands (build, full tests): `run_in_background: true`.
- Never destructive (`rm -rf`, `git reset --hard`, `git push --force`) without explicit authorization.

### 2.3 Git

- **Commit small.** One task = small commit set. Atomic.
- **Clear messages.** `T-G1.0.1: Extend __harness__ metadata with backend field`.
- **Never force-push.**
- **Never skip hooks (`--no-verify`).** Pre-commit and pre-push hooks run `workflows/ci-python.sh` / `workflows/ci-rust.sh` to catch fmt/lint/typecheck/test/build issues before QA sees the code. If a hook fails, FIX THE UNDERLYING ISSUE. Do not bypass. Bypass attempts are visible in git history.
- **Include** `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` on commits.

### 2.4 CI/CD hooks (integrated enforcement)

Two-layer enforcement:

- **Layer 1 — language-level:** `workflows/ci-python.sh` + `workflows/ci-rust.sh` via git hooks (installed by `workflows/install-hooks.sh`). Runs fmt + lint on pre-commit (`quick` mode); full gate on pre-push (`gate` mode). Hooks fire automatically — you don't invoke them.
- **Layer 2 — semantic-level:** the workflow's QA_UNIT. Correctness, scope, architectural alignment.

**When pre-commit rejects your commit:**
- Python: try `ruff format . && ruff check --fix .`
- Rust: try `cargo fmt && cargo clippy --fix --allow-dirty`
- Re-stage, re-commit. Don't use `--no-verify`.

**When pre-push rejects QUEEN's merge:** likely a race condition (file changed between QA pass and push). QUEEN will diagnose — you don't normally trigger this.

**Polyglot note:** hooks auto-detect Python (`pyproject.toml`) and Rust (`Cargo.toml`). A worker touching Python files in a polyglot repo will only trigger the Python pipeline at pre-commit, unless they also changed `Cargo.toml` or `.rs` files.

### 2.5 ZimaBoard

- `root@192.168.50.142`. Historically flaky.
- **Verify first:** `ssh -o ConnectTimeout=5 -o BatchMode=yes root@192.168.50.142 "echo pong"`.
- SSH down = BLOCKER. Don't fake GPU results.
- Useful: `bash bootstrap.sh sync`, `bash bootstrap.sh test`, `bash bootstrap.sh gpu`.

### 2.6 Harness

- Install once: `cd jarvis-harness && pip install -e ".[dev]"`.
- Primary commands:
  - `jarvis-harness test <kernel> --level full` — correctness
  - `jarvis-harness bench <kernel> --save-baseline` — bench + baseline
  - `jarvis-harness bench <kernel>` — bench vs baseline
  - `jarvis-harness test-all` — full regression
  - `jarvis-harness test-all --backend=<name>` — specific backend

### 2.7 Task tracking

- `TaskList` / `TaskUpdate`. Mark assigned task `in_progress` on start, `completed` when acceptance passes.
- **Never mark `completed` if acceptance hasn't passed.**

---

## 3. Reporting format — baseline

Every worker ends with a structured report block. Minimum fields below; role-specific fields in each role doc.

```
==== WORKER REPORT ====
Role: <ROLE>
Task ID: T-<TASK_ID>
Status: COMPLETE | BLOCKED | PARTIAL
Prerequisites verified: yes | no (with reason)

Files changed:
  <list of paths, or "none">

Git commit(s): <SHA list> | not committed — reason

Acceptance command:
  $ <exact command>
  <verbatim output in code block>
Result: PASS | FAIL

Role-specific section:
  <per role doc>

Outstanding issues / things for next worker:
  <honest list; "none" acceptable>
```

**The report is the contract.** Sloppy report → swarm degrades.

---

## 4. Escalation table

| Situation | Action |
|---|---|
| Acceptance command failed | Attempt fix within scope. If >30% of estimate, BLOCKED. |
| Prerequisite not actually complete | Verify by reading deliverables. Report BLOCKED if truly incomplete. |
| Task description ambiguous re: correctness | BLOCKED with specific ambiguity. Do not guess. |
| TODO/ARCH doc has a bug | Fix in small separate commit; note in report. |
| SSH to ZimaBoard fails | Retry twice; then BLOCKED. |
| Scope creep noticed but not yet done | Note in "outstanding"; don't do it. |
| Prior worker did something wrong | Report it; do not silently fix. |
| Pessimistic estimate exceeded | STOP. Report. Diagnose why. |
| Pre-existing TODO/FIXME in unrelated code | Note in "outstanding"; don't fix unless task specifies. |

---

## 5. Git workflow

### 5.1 Before starting

```bash
git status --short
git branch --show-current
```

Uncommitted changes you don't recognize → stop, report.

You should be on the task branch QUEEN created: `task/<TASK_ID>`.

### 5.2 While working

Commit at logical breakpoints. Avoid one giant end-of-task commit.

### 5.3 Before reporting

Re-run the acceptance command from a clean state:

```bash
jarvis-harness test <kernel>
# or whatever the task specifies
```

Paste output into report.

### 5.4 Commit message format

```
T-<TASK_ID>: One-line description

Optional longer explanation.

Acceptance:
  <command>
  <result summary>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## 6. Failure modes — named

### 6.1 The Placeholder Trap

Writing "simplified" code with a comment promising replacement, which never happens. Reference: `ssm_decode.py`.

**Defense:** stubs get `xfail` tests on the same commit.

### 6.2 The Fabricated Report

Claiming a test passed without running. Writing a comment describing an uncomputed result.

**Defense:** every claim → verbatim command output.

### 6.3 Scope Creep

"While I'm here…"

**Defense:** the Do NOT list.

### 6.4 The Silent Disable

Marking a failing test `@skip` instead of fixing it.

**Defense:** QA specifically looks for this.

### 6.5 The Self-Consistency Trap

Validating against a reference with the same bug. Reference: `EVALUATION.md`.

**Defense:** CPU references match a ground truth (llama.cpp, published spec), not the code under test.

### 6.6 The Cleanroom Leak

TESTDEV_BLACKBOX reads a forbidden file "just to understand." Produces tests that encode internals.

**Defense:** QUEEN names forbidden files; QA audits output.

### 6.7 The Lost Context

Forgetting why a decision was made, re-litigating.

**Defense:** INPROGRESS is append-only. Decisions stay visible.

---

## 7. What success looks like

All of:

- Acceptance command was run
- Output matches expected
- Output quoted in report
- No new TODO/FIXME/HACK comments
- No disabled tests
- No scope beyond the task
- Commits small, clear messages
- Report is honest, including about what didn't go as planned

---

## 8. Known gotchas

- **ZimaBoard SSH intermittent** — can drop mid-session
- **Triton JIT requires `hipcc` on PATH** — failure is cryptic
- **PyTorch reports 8.29 GB VRAM** — cosmetic; real ≈ 20 GB
- **Harness test names are long** — use `-k` filtering
- **Build artifacts bloat** — `.gradle/`, `target/`, `build/`, `__pycache__/` are gitignored; don't commit

---

## 9. If in doubt

**Stop. Report. Let QUEEN decide.**

Asking is cheap. Guessing wrong is expensive.

---

*End of protocol. Proceed to your role-specific doc.*
