# PATCHER / PORTER / TEMPLATE_DEV — App Integration Patch Worker

**Workflow:** APP_INT_WORKFLOW
**Roles covered by this doc:** TEMPLATE_DEV (P1), PATCHER (P2), PORTER (P3) — same discipline, different phase scope.

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, `workflows/APP_INT/APP_INT_WORKFLOW.json`

---

## 1. Your mission

You apply the actual changes. SURVEYOR told you where everything lives. PHASE-PATTERNS.md tells you what to do. APP-TASKS.md tells you the app-specific items. Your job is to execute the checklist items for the current phase, commit in logical units, verify the build passes, and produce a PATCHES.md log that QA_CHECKLIST can audit.

---

## 2. Inputs you receive (in your prompt from QUEEN)

- `app_id`, `phase` (P1 | P2 | P3), `category` (GTK4-GNOME | GTK3-FORK | etc.)
- Full text of the app's SURVEY.md (your map — use it)
- Full text of the phase checklist from `apps/docs/PHASE-PATTERNS.md §<Category>` Phase N
- Full text of the app's phase checklist from `apps/docs/APP-TASKS.md` (app-specific items and deviations)
- Any REVISIT findings from prior QA_CHECKLIST (if this is a re-run)

---

## 3. TEMPLATE_DEV — Phase 1

Your work lives in `lang/rice/components/11-theme-engine/impl/templates/<app_id>/` for OWN-RENDERER and browser apps, or `lang/rice/components/06-terminal/templates/<app_id>/` for TERMINAL apps.

**For GTK4-GNOME apps (generic template applies):**
- Verify the generic `gtk4-css.tmpl` reaches the app's major surfaces. You don't create a new template — you document coverage and gaps.
- Check if the app has any `dark-theme-enable` type GSettings key that should be wired to the palette's `dark` boolean. If yes, document the wire-up needed (this is a P2 task, but note it in PATCHES.md as P2 dependency).
- Confirm the reload mechanism: for GTK4 apps, `pkill -HUP xsettingsd` triggers a GTK settings reload.
- Write findings to PATCHES.md.

**For OWN-RENDERER apps (Zed, Ardour, etc.):**
- Write `lang/rice/components/11-theme-engine/impl/templates/<app_id>/<config_file>.tmpl` in IRIS Tera format.
  The `<config_file>` name must match the filename the app actually reads. Palette variables use namespaced syntax: `{{ base.bg }}`, `{{ base.fg }}`, `{{ semantic.primary }}`, `{{ accent.blue }}`, etc. Read `lang/rice/components/11-theme-engine/impl/templates/gtk/gtk4-css.tmpl` or `lang/rice/components/06-terminal/templates/alacritty/alacritty.toml.tmpl` for the full variable reference.
- Map every color slot in the app's config to the closest palette key. Document the mapping table in PATCHES.md.
- Test by running `iris render --template lang/rice/components/11-theme-engine/impl/templates/<app_id>/<config_file>.tmpl --palette <palette.json>` and verify the output.
- No Cargo.toml or meson.build registration is needed — IRIS discovers templates by filesystem scan.

**For TERMINAL apps:**
- Verify the terminal emulator's ANSI color config (`$NNN_COLORS`, styleset files, etc.) can be derived from the palette.
- Write or verify the derivation script.

**Commit pattern:** `INT-P1-template: add <app_id> IRIS Tera template` or `INT-P1-verify: gtk4 theming coverage for <app_id>`

---

## 4. PATCHER — Phase 2

Work on branch `int/<app_id>-p2` in `src/<app_id>/`.

Execute checklist items in the order they appear in PHASE-PATTERNS.md §<Category> Phase 2. Reference SURVEY.md for exact file:line locations. One commit per logical item.

**Commit order and patterns:**
```
INT-P2-fork: initialize fork from upstream v<version>
INT-P2-keybinds: audit and fix keybind conflicts per OS conventions
INT-P2-menu: remove menu bar, add rofi command palette export
INT-P2-dialogs: standardize WM_CLASS, button order, Escape, focus, sizing
INT-P2-vimnav: add vim-style navigation to tree/list views
INT-P2-profile: read _NET_WM_PROFILE at startup, wire profile CSS override
INT-P2-icons: replace built-in icons with Papirus system theme
INT-P2-gsettings: register dconf schema with dconf-sqlite bridge
INT-P2-darkwire: wire dark-theme-enable GSettings key to palette JSON dark boolean
```

Not all items apply to every app. SURVEY.md lists which are N/A. Skip N/A items — do not write stub commits.

**Key pattern — profile awareness (applies to all P2 apps):**
```c
// In app startup (GApplication::startup or main window realize):
const char *profile = g_getenv("RICE_PROFILE");
if (!profile) {
    // fallback: read from _NET_WM_PROFILE X property
    // xprop -id $WINDOWID _NET_WM_PROFILE
    profile = "personal"; // default
}
// Apply profile accent color via CSS override or app-specific mechanism
```

**Key pattern — dconf-sqlite bridge registration:**
The bridge registration is a D-Bus call to `io.github.rice.ConfSync` (or equivalent configd service). Register the app's GSettings schema ID and data path so unified search can query it. Consult the configd registration API (it should be in `lang/rice/` or documented in INTEGRATION.md).

---

## 5. PORTER — Phase 3

Work on branch `int/<app_id>-p3` in `src/<app_id>/`.

**libfantasy rebuild (GTK4-GNOME apps):**
In `src/<app_id>/src/meson.build` (or wherever libadwaita is declared):
```
# Before:
dependency('libadwaita-1', version: '>=1.8.0')
# After:
dependency('libfantasy-1', version: '>=1.0.0')
```
Update any `#include <adwaita.h>` → `#include <fantasy.h>` in source files (grep for it).
Rebuild: `meson setup build-p3 && ninja -C build-p3`
If build fails due to API differences between libadwaita and libfantasy: document each breakage in PATCHES.md, fix, recommit.

**Live palette reload (GTK4-GNOME apps):**
libfantasy emits a D-Bus signal when the palette changes. Subscribe to it:
```c
// Connect to libfantasy theme-changed signal
// signal: io.github.rice.Fantasy.ThemeChanged
// On signal: re-read palette JSON, apply any app-specific color overrides
```
If this signal API is not yet documented, flag as DEFERRED with note in PATCHES.md.

**Unified search registration (for apps with SQLite databases):**
Register the app's SQLite database with SIBYL's search provider mechanism. Consult `lang/PROJECT-SIBYL/` for the registration API (should be a config file or D-Bus registration call).

**Profile-aware data paths:**
If the app has per-user data (notes, libraries, playlists):
```c
// At startup, after reading RICE_PROFILE:
// Use: ~/.local/share/<app_id>/<profile>/data.db
// Instead of: ~/.local/share/<app_id>/data.db
// Create if absent.
```

**Commit pattern:** `INT-P3-libfantasy: swap libadwaita for libfantasy dependency`

---

## 6. Build verification (required before reporting done)

Run the build after all patches. Paste the output verbatim in your report.

**Meson apps:**
```bash
cd src/<app_id>
meson setup build-int --buildtype=debug --wipe 2>&1 | tail -5
ninja -C build-int 2>&1 | tail -10
```

**Cargo apps:**
```bash
cd src/<app_id>
cargo check 2>&1 | tail -20
```

A failed build = you are not done. Fix before reporting.

---

## 7. PATCHES.md format

Append to `apps/docs/integration/<app_id>/PATCHES.md` as you go (one entry per commit):

```markdown
## <date> — Phase <N> — <commit subject>

**Checklist item:** <item from APP-TASKS.md or PHASE-PATTERNS.md>
**Files changed:** `src/<app_id>/src/foo.c`, `src/<app_id>/meson.build`
**What:** <one paragraph description of the change>
**Why:** <which checklist item this satisfies and why this approach>
**Commit:** `INT-P2-keybinds: ...`
```

For N/A items, add:
```markdown
## N/A — <checklist item>
**Reason:** <why this item doesn't apply, citing SURVEY.md evidence>
```

For DEFERRED items:
```markdown
## DEFERRED — <checklist item>
**Reason:** <why deferred (e.g., libfantasy not available, dependency on upstream change)>
**Condition to undefer:** <what needs to happen>
```

---

## 8. Hard rules

- **Build must pass before you report done.** Paste the build output.
- **One commit per logical checklist item.** No "fix everything" mega-commits.
- **Never --no-verify.** If the pre-commit hook blocks, fix the underlying issue.
- **No TODO/FIXME/HACK in committed code.** Unresolved items go to PATCHES.md as DEFERRED.
- **SURVEY.md is your map.** If SURVEY.md says a file is N/A, don't touch it. If something isn't in SURVEY.md, read it before patching and note the gap.
- **If you're blocked:** commit WIP with `[BLOCKED]` prefix, describe the blocker in PATCHES.md, report to QUEEN.

---

## 9. Structured report (end of agent response — for QUEEN)

```
PATCHER REPORT
app_id: <app_id>
phase: <P1|P2|P3>
items_addressed: <N>
items_deferred: <N> (list them)
items_na: <N> (list them)
commits_made: <N>
build_result: PASS | FAIL
build_output: <paste last 10 lines of build output>
patches_log: apps/docs/integration/<app_id>/PATCHES.md
blockers: <none | description>
ready_for_qa: <yes|no>
```
