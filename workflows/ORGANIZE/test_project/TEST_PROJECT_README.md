# ORGANIZE Test Project

This directory is a deliberately-messy small project used to exercise ORGANIZE workflow walkthroughs.

**Intentional state:**
- No `.organize.json` — so BOOTSTRAP runs cleanly
- Mix of canonical (`src/`, `tests/`, `docs/`) and messy (root-level cruft, `backup/`, `tmp/`) structure
- Applies to: `mixed-research` template
- ~20 files designed to exercise all 6 TRIAGE verdicts

## File inventory

### Canonical (well-placed, should be KEEP_IN_PLACE)
- `README.md` — project description
- `src/main.py` — simulation entry point
- `src/utils/helpers.py` — utility functions
- `src/utils/__init__.py` — package init
- `src/__init__.py` — package init
- `tests/test_main.py` — pytest tests
- `tests/__init__.py` — package init
- `docs/overview.md` — project overview
- `pyproject.toml` — Python project config

### Messy (root-level cruft, should generate triage verdicts)
- `scratch.py` — obvious scratch content; should QUARANTINE:.delete
- `notes2.md` — dated old notes matching `notes[0-9]*.md`; should QUARANTINE:.archive
- `old_design.md` — contains "SUPERSEDED" marker; should QUARANTINE:.archive
- `tmp_data.csv` — temporary data file matching `tmp*`; should QUARANTINE:.delete
- `untitled.md` — unclear purpose; should ASK_USER

### Cruft directories
- `backup/old_code.py` — clearly archived code; should QUARANTINE:.archive
- `tmp/experiment.py` — obviously temporary; should QUARANTINE:.delete

### Root invariants (protected — ORGANIZE never touches)
- `.claude/` — with `.gitkeep` placeholder
- `.claude-flow/` — with `.gitkeep` placeholder
- `.hive-mind/` — with `.gitkeep` placeholder
- `.mcp.json` — MCP configuration (empty JSON)

## Purpose

Walk through BOOTSTRAP_WALKTHROUGH.md and MAINTENANCE_WALKTHROUGH.md using this project
as the target. The files are designed so that all 6 TRIAGE verdict types are exercised
in a single circuit.
