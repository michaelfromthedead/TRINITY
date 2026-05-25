# SDLC WORKFLOW — Cron Job Prompt

**Purpose:** Execute one SDLC development cycle on the next pending subtask.
**Location:** workflows/SDLC/
**State:** workflows/SDLC/SDLC_STATE.json
**Sprint:** docs/sprints/PHASE_A_SPRINT.md

---

## PHASE 0: KNOWLEDGE INJECTION (MANDATORY)

Before ANY work, read these files IN FULL:

1. **workflows/SDLC/SDLC_WORKFLOW.json** — Full workflow spec (320 lines)
2. **workflows/SDLC/WORKER_DEV.md** — DEV worker contract
3. **workflows/SDLC/WORKER_TESTDEV_WHITEBOX.md** — Whitebox tester contract
4. **workflows/SDLC/WORKER_TESTDEV_BLACKBOX.md** — Blackbox tester contract
5. **workflows/SDLC/WORKER_QA_JUNIOR.md** — Junior QA contract
6. **workflows/SDLC/WORKER_QA_SENIOR_SANITY.md** — Senior QA sanity contract
7. **workflows/SDLC/WORKER_QA_SENIOR_FINAL.md** — Senior QA final contract
8. **workflows/SHARED/WORKER_QUEEN.md** — QUEEN orchestrator contract
9. **workflows/SHARED/WORKER_PROTOCOL.md** — Universal worker protocol
10. **docs/sprints/PHASE_A_SPRINT.md** — Current sprint details

---

## PHASE 1: READ STATE

```bash
cat workflows/SDLC/SDLC_STATE.json
```

Parse the state:
- Find `next_action` — what needs to happen
- Find tasks with status `PENDING` or `COMPROMISED`
- Find first subtask within that task with status `PENDING`

If all tasks are `DONE`: Report "PHASE A COMPLETE" and stop.

If `audit_required` is true: Run AUDIT mode instead of normal development.

---

## PHASE 2: AUDIT MODE (if audit_required)

For COMPROMISED tasks:

1. **Read the compromised code** — every line of the implementation
2. **Compare against sprint checklist** — what was implemented vs what was specified
3. **Run existing tests** — `uv run python -m pytest <test_file> -v`
4. **Run syntax check** — `uv run python -m py_compile <file>`
5. **Document gaps** — what's missing, what's wrong, what needs fixing
6. **Update state** — mark subtask as AUDITED with findings

After audit, decide:
- If code is acceptable: Mark DONE
- If code needs fixes: Mark FIX_REQUIRED with specific issues
- If code needs rewrite: Mark REWRITE_REQUIRED

---

## PHASE 3: DEVELOPMENT MODE (normal flow)

For PENDING subtasks:

### 3a. DEV Phase

Spawn DEV worker:
- Read the subtask specification from sprint doc
- Read existing code in the target path
- Implement the subtask
- Write code using Python 3.13 (use `uv run python`)

### 3b. TEST_UNIT (Parallel)

Spawn in parallel:
- **TESTDEV_WHITEBOX**: Write unit tests with internal knowledge
- **TESTDEV_BLACKBOX**: Write integration tests from external perspective

Run tests: `uv run python -m pytest tests/<relevant>/ -v`

### 3c. QA_UNIT (Sequential)

1. **QA_JUNIOR**: Basic syntax check, import verification
   ```bash
   uv run python -m py_compile <file>
   uv run python -c "from <module> import *; print('OK')"
   ```

2. **QA_SENIOR_SANITY**: Quick sanity review
   - Does code match specification?
   - Any obvious issues?

3. **QA_SENIOR_FINAL**: Full acceptance review
   - All acceptance criteria met?
   - Tests pass?
   - Code quality acceptable?

### 3d. VERDICT

- **GREEN_LIGHT**: Subtask complete, update state to DONE
- **FIX**: Minor issues, loop back to DEV with specific fixes
- **REWRITE**: Major issues, loop back to DEV for full rewrite
- **ESCALATE**: Blocked, needs human decision

---

## PHASE 4: UPDATE STATE

After completing a subtask:

1. Update `workflows/SDLC/SDLC_STATE.json`:
   - Set subtask status to new value
   - Add entry to history array
   - Update next_action and next_subtask

2. Report completion:
   ```
   SDLC CYCLE COMPLETE
   Task: <task_name>
   Subtask: <subtask_name>
   Result: <GREEN_LIGHT|FIX|REWRITE|ESCALATE>
   Next: <next_subtask or "TASK COMPLETE">
   ```

---

## HARD RULES

From SDLC_WORKFLOW.json:
- `test_unit_is_parallel: true` — WHITEBOX and BLACKBOX run together
- `qa_unit_is_sequential: true` — JUNIOR → SANITY → FINAL in order
- `no_greenlight_without_full_qa_unit: true` — must pass all QA
- `loop_limit: 3` — max 3 FIX/REWRITE cycles before ESCALATE

**CRITICAL: Python 3.13 Required**
```bash
uv run python --version  # Must show 3.13.x
uv run python -m pytest  # NOT just "pytest"
uv run python -m py_compile  # NOT just "py_compile"
```

---

## OUTPUT FORMAT

Each cycle ends with a state update and report:

```json
{
  "cycle": "2026-05-23T01:00:00Z",
  "task": "T1",
  "subtask": "T1.2",
  "action": "DEV",
  "result": "GREEN_LIGHT",
  "tests_passed": true,
  "qa_verdict": "APPROVED",
  "next": "T1.3"
}
```

---

*Created: 2026-05-23*
*For: Phase A Sprint Development*
