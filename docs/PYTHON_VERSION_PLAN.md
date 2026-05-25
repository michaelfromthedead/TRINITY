# TRINITY Python Version Plan

**Created:** 2026-05-22
**Status:** Active
**Target:** Python 3.13.x (statically linked binary)

---

## ⚠️ CRITICAL WARNING

```
┌─────────────────────────────────────────────────────────────────┐
│  THIS PROJECT REQUIRES PYTHON 3.13                              │
│                                                                 │
│  System default: 3.14.4  ← DO NOT USE                           │
│  Project target: 3.13.x  ← USE THIS                             │
│                                                                 │
│  ALWAYS prefix Python commands with: uv run                    │
│                                                                 │
│  ✗ python script.py          # WRONG - uses 3.14               │
│  ✓ uv run python script.py   # CORRECT - uses 3.13             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current State

| Environment | Python Version | Status |
|-------------|----------------|--------|
| System default | 3.14.4 | Active (global) |
| TRINITY project | 3.13.13 | **Pinned via uv** |
| Custom binary | 3.13.x | Statically linked target |

## Why Python 3.13?

TRINITY uses a **custom statically-linked Python 3.13 interpreter** as the runtime target. This is a deliberate architectural decision:

1. **Stability** — 3.13 is a stable release with known behavior
2. **Static linking** — Eliminates runtime dependency on system Python
3. **Embedding** — The interpreter is embedded in the Rust/C++ binary
4. **Reproducibility** — Ensures identical behavior across all deployments

## Configuration

### Directory Pin (`.python-version`)

```
3.13
```

This file tells `uv`, `pyenv`, and other tools to use Python 3.13 when working in this directory.

### uv Setup

```bash
# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.13
uv python install 3.13

# Pin this directory
uv python pin 3.13

# Verify
uv run python --version  # Should show 3.13.x
```

### Virtual Environment (Recommended)

```bash
# Create venv with pinned Python
uv venv --python 3.13

# Activate
source .venv/bin/activate

# Verify
python --version  # 3.13.x
```

## Compatibility Notes

### 3.13 vs 3.14 Differences

| Feature | 3.13 | 3.14 |
|---------|------|------|
| GIL | Optional (PEP 703 experimental) | Free-threaded default |
| JIT | Experimental tier-2 | Production tier-2 |
| Typing | Full | Extended |
| Match patterns | Full | Extended |

### Code Compatibility

The TRINITY codebase (~600,000 lines) was written targeting 3.13 features:

- **Type hints** — Use 3.13-compatible syntax
- **Match statements** — PEP 634 (available since 3.10)
- **Dataclasses** — `slots=True` (3.10+)
- **TypedDict** — `Required`/`NotRequired` (3.11+)

**Avoid** 3.14-only features:
- `type` statement soft keywords
- Enhanced error messages (runtime-only, not syntax)
- New `typing` module additions

## GRANDPHASE2 Bridge Integration

The statically-linked Python 3.13 binary is the foundation for:

1. **PyO3 bindings** — Rust ↔ Python FFI via libpython3.13
2. **Embedded interpreter** — Python scripts execute inside the engine
3. **Hot reload** — Python modules can be reloaded without restarting
4. **WGSL codegen** — Python generates shader code at build time

### Build Chain

```
┌─────────────────────┐
│   TRINITY Python    │
│   (~600K lines)     │
└──────────┬──────────┘
           │ imported by
           ▼
┌─────────────────────┐
│  libpython3.13.a    │
│  (static library)   │
└──────────┬──────────┘
           │ linked into
           ▼
┌─────────────────────┐
│  renderer-backend   │
│  (Rust + PyO3)      │
└─────────────────────┘
```

## Verification Commands

```bash
# Check pinned version
cat .python-version

# Check uv's understanding
uv python list --only-installed

# Run with correct Python
uv run python --version

# Test import
uv run python -c "import sys; print(sys.version_info)"
```

## Migration Path

If upgrading to a newer Python in the future:

1. Update the statically-linked binary first
2. Test all 600K lines against new version
3. Update `.python-version`
4. Rebuild PyO3 bindings
5. Run full test suite

---

*This document is authoritative for TRINITY Python version decisions.*
