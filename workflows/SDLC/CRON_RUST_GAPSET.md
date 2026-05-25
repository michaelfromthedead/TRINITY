# RUST GAPSET CRON — Worker Pool SDLC

**Purpose:** Execute GAPSET SDLC for Rust/compiler work (GAPSET_2_FRAME_GRAPH).
**Frequency:** Every 10 minutes (`7,17,27,37,47,57 * * * *`)
**Model:** Worker pool — 4 slots, any stage (DEV/TEST/QA/ACCEPTANCE).

---

## SYSTEMIC FIXES (2026-05-23 — sister-swarm lessons)

### Fix 1: Disk-first every cycle (NEVER respond from memory)
- Every cron tick: READ the tracker file, COUNT running agents via `mcp__ruflo__agent_list`, CHECK build status via `cargo test`
- NEVER report "pool full" without verifying. NEVER assume agents are still running.

### Fix 2: AgentDB persistence (immune to file reverts)
- Store pool state in AgentDB `memory_store` under namespace `gapset-sdlc`
- Key: `pool-state` — tracks active agents, in-flight tasks, stage
- If tracker file or cron file is reverted by hooks, AgentDB state survives
- On each cycle: read AgentDB state first, reconcile with disk

---

## WORKER POOL LOGIC

```
POOL SIZE: 4 workers max.
HARD CAP: 8 agents total across ALL crons.

Each cycle:
1. READ from disk (NOT memory):
   - Check mcp__ruflo__agent_list for currently-running agents
   - Read memory_store("pool-state", namespace="gapset-sdlc") for in-flight tasks
   - Read GAPS_SDLC_TODO.md for current gapset
2. Count actual empty slots = 4 minus verified-running agents
3. Process completions → advance tasks to next stage
4. Fill empty slots with any unblocked stage (DEV/TEST/QA/ACCEPTANCE)
5. Store updated pool-state back to AgentDB
6. Report with actual numbers, not memory

PIPELINE PER TASK:
  DEV → WHITEBOX ∥ BLACKBOX → JUNIOR_QA → SANITY → FINAL → GREEN_LIGHT
```

---

## INVOCATION PROMPT

```
GAPSET SDLC — Worker Pool Cycle (4 slots)

**HARD CAP: 8 agents max total. Never exceed.**
**DISK-FIRST: Read files, count running agents, verify before reporting.**

1. VERIFY: Run mcp__ruflo__agent_list to count actually-running agents
2. RECONCILE: Read memory_store("pool-state", namespace="gapset-sdlc")
   to recover in-flight task state (immune to file reverts)
3. READ docs/gap_sets/GAPS_SDLC_TODO.md — find current gapset
4. COUNT empty slots = 4 minus verified-running agents
5. FILL slots in priority order (QA > TEST > DEV) for unblocked tasks
6. SPAWN all workers in ONE message, run_in_background:true
7. STORE updated pool-state via memory_store("pool-state", ...)
8. COMMIT any disk changes from prior cycle (wait for lock, never rm -f)
9. REPORT with actual numbers: "POOL: N active. Verified via agent_list.
   Completions: A. Stages advanced: X. Build: PASS/FAIL. Tests: P/F."

Reference: workflows/SDLC/CRON_RUST_GAPSET.md
```

---

## SPAWN PATTERNS

### DEV (1 slot)
```
Agent(description:"DEV <TASK>", name:"dev-<task>", subagent_type:"coder",
  run_in_background:true, prompt:"SDLC DEV for <TASK>. CODE ONLY. No tests.")
```

### WHITEBOX + BLACKBOX (2 slots, parallel)
```
Agent(description:"WHITEBOX <TASK>", name:"wb-<task>", subagent_type:"tester",
  run_in_background:true, prompt:"WHITEBOX tests for <TASK>. Full source access.")
Agent(description:"BLACKBOX <TASK>", name:"bb-<task>", subagent_type:"tester",
  run_in_background:true, prompt:"BLACKBOX tests for <TASK>. CLEANROOM.")
```

### JUNIOR_QA (1 slot)
```
Agent(description:"JUNIOR <TASK>", name:"junior-<task>", subagent_type:"tester",
  run_in_background:true, prompt:"JUNIOR_QA for <TASK>. Hypercritical.")
```

### SANITY (1 slot)
```
Agent(description:"SANITY <TASK>", name:"sanity-<task>", subagent_type:"reviewer",
  run_in_background:true, prompt:"SENIOR_QA_SANITY for <TASK>. Judicial stance.")
```

### FINAL (1 slot)
```
Agent(description:"FINAL <TASK>", name:"final-<task>", subagent_type:"reviewer",
  run_in_background:true, prompt:"SENIOR_QA_FINAL for <TASK>. Independent verdict.")
```

---

## GAPSET ORDER
1→2→3(SKIP)→4→5→6→7→8→9→10→11→12→13→14→15→16→17→18→19→20

## HARD RULES
- DISK-FIRST: never report from memory
- Pool: 4 workers per cycle
- HARD CAP: 8 agents max across ALL running tasks
- AgentDB backup: store pool-state every cycle
- QA stages are SEQUENTIAL
- WHITEBOX ∥ BLACKBOX always together (2 pool slots)
- Never skip QA_UNIT for any task

### GIT COMMIT RULE (CONCERNING_PROGRESS.md §8.4)
- **NEVER commit while agents are running.** Verify with agent_list first.
- **NEVER run git commit in background.** Synchronous only.
- **NEVER rm -f .git/index.lock blindly.** Only remove if NO git process holds it.
- **Commit phase is AFTER agent completions, BEFORE spawning new agents.**
- Pattern: process completions → agent_list==0 → commit → spawn new cycle
- If agents still running → skip commit, try next cycle

---

## AUTONOMY REQUIREMENTS

For the cron to run without permission prompts, these MUST be in `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(git *)",
      "Bash(cargo *)",
      "Bash(sed *)",
      "Bash(cp *)",
      "Bash(mv *)",
      "Bash(rm .git/index.lock)",
      "Bash(for *)",
      "Bash(while *)",
      "Bash(find *)",
      "Bash(ls *)",
      "Bash(head *)",
      "Bash(tail *)",
      "Bash(wc *)",
      "Bash(cat *)",
      "Bash(echo *)",
      "Bash(ps *)",
      "Bash(sort *)",
      "Bash(uniq *)",
      "Bash(cd * && git *)",
      "Bash(cd * && cargo *)",
      "mcp__ruflo__memory_*",
      "mcp__ruflo__agent_*",
      "mcp__ruflo__swarm_*",
      "mcp__ruflo__hooks_route",
      "CronCreate",
      "CronDelete",
      "CronList"
    ]
  }
}
```

**Without these permissions**, every Agent spawn, every git commit, every cargo test, and every memory_store will trigger an approval prompt — breaking autonomy.

**Check:** Run `cat .claude/settings.local.json | python3 -m json.tool` to verify the allowlist is active.

*Updated: 2026-05-23 — Worker pool + systemic fixes + autonomy config*
