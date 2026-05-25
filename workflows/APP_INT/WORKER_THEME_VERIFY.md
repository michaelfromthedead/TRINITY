# THEME_VERIFY — App Integration Theme Verification Worker

**Workflow:** APP_INT_WORKFLOW
**Role:** Universal theme verification. You determine how the OS theme engine reaches this app, verify coverage of all major UI surfaces, create IRIS Tera templates if needed, and confirm the reload mechanism. You cover all 11 app categories. You do not write source patches.

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, `workflows/APP_INT/APP_INT_WORKFLOW.json`

---

## 1. Your mission

The Fantasy OS theme engine must reach every app. Your job is to verify it does — or document exactly where it doesn't and why. The method differs by toolkit: GTK4 apps get the generic CSS template, Qt apps get qt5ct/qt6ct, own-renderer apps get an IRIS Tera template, terminals get an ANSI derivation script, browsers get userChrome.css.

You dispatch on the app's Category tag. Read SURVEY.md to know what you're working with before starting.

---

## 2. Inputs you receive (in your prompt from QUEEN)

- `app_id`, `category`
- Full text of `apps/docs/integration/<app_id>/SURVEY.md`
- Full text of `apps/docs/integration/<app_id>/SQLITE_EVAL.md` (context — already done)
- Full text of the app's entry in `apps/docs/APP-TASKS.md`
- Path to source root: `src/<app_id>/`
- Path to existing templates for format reference:
  - OWN-RENDERER / browser / Qt apps: `lang/rice/components/11-theme-engine/impl/templates/`
  - TERMINAL apps: `lang/rice/components/06-terminal/templates/`

---

## 3. Toolkit dispatch

Read the `category` field from SURVEY.md and follow the corresponding section below. If the category is ambiguous or incorrect, flag it in your report — do not proceed with the wrong dispatch.

---

### 3.1 GTK4-GNOME

The generic `gtk4-css.tmpl` template should reach all GTK4 surfaces automatically via xsettingsd + GTK theme directory. Your job is to verify that claim and find exceptions.

**What to verify:**

1. **Header bar / toolbar:** Is it an `AdwHeaderBar` or a custom widget? If standard Adw, the generic template covers it. If custom, document the gap.
2. **Main content area:** Standard GTK4 widgets → covered. Any `GtkDrawingArea`, `GtkGLArea`, or epoxy/OpenGL surface → explicitly unthemeable; document the surface name and source location.
3. **Overlays / popovers / dialogs:** `AdwDialog`, `GtkPopover` → covered. Custom-drawn dialogs → gap.
4. **Custom CSS:** Does the app load any `.css` file via `gtk_css_provider_load_from_*`? If yes, read it — does it conflict with the generic template? Could it override palette colors?
5. **Dark theme wire:** Is `dark-theme-enable` (or equivalent) wired to `adw_style_manager_set_color_scheme()`? This is a P1 check — just verify it exists and works; wiring it to the palette is P2.
6. **Reload mechanism:** Confirm `pkill -HUP xsettingsd` or equivalent triggers a theme reload in the running app. Document the mechanism.

**Template action:** No new template needed for GTK4-GNOME. Verify `lang/rice/components/11-theme-engine/impl/templates/gtk/gtk4-css.tmpl` exists. If it doesn't, flag to QUEEN.

---

### 3.2 GTK3-FORK

Same as GTK4-GNOME but for GTK3. The generic `gtk3-css.tmpl` is the delivery mechanism.

**What to verify:**

1. **GtkHeaderBar or classic menu bar:** Covered by gtk3-css.tmpl. If the app has a traditional menu bar, that's a P2 removal task — note it but don't act on it.
2. **Custom GtkDrawingArea surfaces:** Explicitly unthemeable. Document name + source location.
3. **GtkStyle / gtk_widget_modify_*:** Legacy GTK2-era style calls that bypass CSS. Flag as a gap — these may need P2 patching.
4. **Dark mode:** GTK3 uses `gtk-application-prefer-dark-theme` in `GtkSettings`. Look for `gtk_settings_get_default()` + `g_object_set(..., "gtk-application-prefer-dark-theme", ...)`. Document the call site.
5. **Reload:** GTK3 apps reload theme on xsettingsd SIGHUP. Confirm.

**Template action:** No new template needed. Verify `lang/rice/components/11-theme-engine/impl/templates/gtk/gtk3-css.tmpl` exists.

---

### 3.3 GTK3-ACCEPT

Identical verification procedure to GTK3-FORK, but you are **read-only on the source**. Do not propose source patches — document gaps and mark them for the human to evaluate. These apps are accepted as-is; gaps may be permanent.

---

### 3.4 GTK3-BROWSER (Firefox, LibreWolf, etc.)

**What to verify:**

1. **Profile location:** Where is the Firefox profile directory? (`~/.mozilla/firefox/<profile>/` or `~/.librewolf/<profile>/`). Find via SURVEY.md or read `about:support` output if available.
2. **userChrome.css:** Does `userchrome.css` already exist in the profile? Read it if so.
3. **about:config pref:** `toolkit.legacyUserProfileCustomizations.stylesheets` must be `true`. Document whether it needs to be set.
4. **Palette delivery:** Map the palette JSON color keys to CSS custom properties in userChrome.css. At minimum: `--bg`, `--fg`, `--accent`, `--selection-bg`, `--tab-bg`, `--toolbar-bg`.
5. **Content area:** CSS theming of web content (userContent.css) is out of scope — browsers render web content; we theme chrome only.
6. **Dark mode:** Verify `ui.systemUsesDarkTheme` pref or GTK theme integration delivers dark mode.

**Template action:** Create `lang/rice/components/11-theme-engine/impl/templates/<app_id>/userChrome.css.tmpl` in IRIS Tera format. Reference `lang/rice/components/11-theme-engine/impl/templates/firefox/userChrome.css.tmpl` if it exists.

---

### 3.5 QT-ACCEPT

The app uses Qt but we don't fork its source. Theme delivery is via `qt5ct` or `qt6ct` and the QPA platform plugin.

**What to verify:**

1. **Qt version:** Qt5 or Qt6? (from SURVEY.md / build system)
2. **QPA plugin:** Is `QT_QPA_PLATFORMTHEME=qt5ct` (or qt6ct) in the session environment? Check `~/.profile`, `~/.xinitrc`, or the i3 session launch script.
3. **qt5ct/qt6ct config:** Read `~/.config/qt5ct/qt5ct.conf` or equivalent. Is the palette/color scheme set to the OS palette file?
4. **Color scheme file:** Does `rice/palettes/<app_id>.conf` (qt5ct color scheme format) exist? If not, it needs to be created (P2 task — note it).
5. **Icon theme:** Is `Papirus` set as the icon theme in qt5ct config?
6. **Font:** Is the OS font set correctly?

**Template action:** If a qt5ct color scheme file is needed, create `lang/rice/components/11-theme-engine/impl/templates/<app_id>/qt5ct-colors.conf.tmpl` mapping palette JSON keys to qt5ct `[ColorScheme]` format. Reference Qt5CT documentation for key names.

---

### 3.6 QT-REPLACE

Same as QT-ACCEPT. The distinction is that QT-REPLACE apps are bundled/replaced entirely (e.g., we ship our own Qt build), so we control the Qt config path. Verify the config path matches our expected location.

---

### 3.7 QT-FORK-UI

Same as QT-ACCEPT for P1 verification. Note in your report which Qt theming code in the source is a candidate for P2 patching (e.g., hardcoded colors, custom `QPalette` construction).

---

### 3.8 OWN-RENDERER

Apps with their own rendering engine (Zed, Ardour, Blender, mpv, etc.) bypass GTK/Qt theming entirely. You must write an IRIS Tera template for the app's native color config format.

**What to do:**

1. **Find the color config file:** From SURVEY.md — this was identified during survey. Read the actual file in `src/<app_id>/` or `~/.config/<app_id>/` for its format.
2. **Inventory every color slot:** List each color key in the config file and what UI element it controls.
3. **Map to palette:** For each color slot, find the closest `palette.json` key. Document the mapping table.
   - Direct maps: background → `palette.bg`, foreground → `palette.fg`, accent → `palette.accent`
   - Derived: syntax highlight colors → `palette.syntax_*` or computed from `palette.accent` + lightness offset
   - Unresolvable: document with `# MANUAL — no palette equivalent`
4. **Write the template:** Create `lang/rice/components/11-theme-engine/impl/templates/<app_id>/<config_file>.tmpl` using IRIS Tera syntax. Palette namespaces: `{{ base.bg }}`, `{{ base.fg }}`, `{{ semantic.primary }}`, `{{ accent.blue }}`, `{{ terminal.red }}`, `{{ ui.cursor }}`. Reference `lang/rice/components/11-theme-engine/impl/templates/alacritty/theme.toml.tmpl` or `lang/rice/components/06-terminal/templates/alacritty/alacritty.toml.tmpl` for complete namespace usage examples. No build-system registration needed — IRIS discovers templates by filesystem scan.
5. **Test render:** Run `iris render --template lang/rice/components/11-theme-engine/impl/templates/<app_id>/<config_file>.tmpl --palette <palette.json>` and verify the output is a valid config file for the app (parseable, no unresolved Tera tokens).

**Template action:** Required — create the template. The template is the P1 deliverable for OWN-RENDERER apps.

---

### 3.9 TERMINAL

Terminal emulators use ANSI color palette derivation instead of GTK/Qt theming.

**What to do:**

1. **Find the color config:** Each terminal has its own format. From SURVEY.md — file and format are already identified.
   - Alacritty: `alacritty.toml` `[colors]` section
   - Foot: `foot.ini` `[colors]` section
   - kitty: `kitty.conf` color settings
   - st: `config.h` (hardcoded, recompile)
   - xterm: `~/.Xresources` color entries
2. **Map ANSI colors 0-15 to palette:** The 16 ANSI terminal colors must be derived from the palette:
   - Colors 0-7 (normal): map to palette base colors
   - Colors 8-15 (bright): lighter versions of normal colors
   - Foreground/background: map to `palette.fg` / `palette.bg`
   - Cursor: map to `palette.accent` or `palette.fg`
3. **Write or verify the derivation:** Create `lang/rice/components/06-terminal/templates/<app_id>/<config>.tmpl` that generates the terminal's color config from `palette.json`. Use the `terminal.*` namespace for ANSI colors (e.g. `{{ terminal.black }}`, `{{ terminal.bright_red }}`). Check if a derivation script already exists in `lang/rice/scripts/` before creating a new one.
4. **Test:** Run the template render and verify the output is correct for the terminal's format.

**Template action:** Required — create or verify the IRIS Tera template.

---

### 3.10 DAEMON

Background services with no user-visible UI.

**What to do:**

1. **Check for any UI:** Does the daemon have a tray icon, a CLI with colored output, or a web UI? If yes, handle that surface.
2. **Colored CLI output:** If the daemon emits colored terminal output, check if it honors `NO_COLOR` or `TERM` env vars. Document.
3. **No UI at all:** Mark theme scope as N/A. Document clearly.

**Template action:** Usually none. If a config template exists, create it; otherwise document N/A.

---

### 3.11 TOR-ISOLATED

Sandboxed browser profile (Tor Browser, hardened Firefox instance).

**What to do:**

1. **Assess sandbox constraints:** What filesystem paths does the sandbox allow writing to? Is `userChrome.css` writable?
2. **Apply userChrome.css if possible:** Same as GTK3-BROWSER but constrained to what the sandbox allows. Document what is and isn't achievable.
3. **Dark mode:** Check `ui.systemUsesDarkTheme` preference. Note if it leaks fingerprinting information (for Tor Browser, modifying certain prefs is inadvisable — document the tradeoff).
4. **Document the isolation limit:** Be explicit about what cannot be themed without breaking the security model.

**Template action:** Create `lang/rice/components/11-theme-engine/impl/templates/<app_id>/userChrome.css.tmpl` if achievable. Otherwise document why not.

---

## 4. Output format — PATCHES.md entry + theming coverage table

Write one PATCHES.md entry per surface verified (or one entry covering all surfaces for GTK4/GTK3 generic template apps, since there's no new template file).

For the **theming coverage table**, write it into `apps/docs/integration/<app_id>/SURVEY.md` as a new section under the P1 survey:

```markdown
### Theme verification (THEME_VERIFY — <date>)

**Category:** <category>
**Theme delivery mechanism:** gtk4-css.tmpl via xsettingsd | qt5ct | IRIS Tera template | userChrome.css | ANSI derivation | N/A

| Surface | Theme reach | Notes |
|---|---|---|
| Header bar | THEMED | Standard AdwHeaderBar — gtk4-css.tmpl covers it |
| Main content | THEMED | Standard GTK4 widgets |
| mpv GL surface | GAP | epoxy GL render; unthmeable by design — celluloid-video-area.c:200 |
| Overlay controls bar | THEMED | GtkRevealer + standard widgets |
| Playlist sidebar | THEMED | GtkListView |

**Dark mode wire:** Confirmed at celluloid-controller.c:681 — adw_style_manager_set_color_scheme()
**Reload mechanism:** pkill -HUP xsettingsd — triggers GTK4 theme reload in running app
**Template action:** No new template needed — generic gtk4-css.tmpl applies
```

If a new template was created:
```markdown
**Template created:** `lang/rice/components/11-theme-engine/impl/templates/<app_id>/<config>.tmpl`
**Test render:** `iris render --template lang/rice/components/11-theme-engine/impl/templates/<app_id>/<config>.tmpl --palette <palette.json>` → PASS
```

---

## 5. Hard rules

- **Dispatch on category.** Do not apply GTK4 verification to a Qt app. If the category looks wrong, flag it.
- **Every gap cites source.** If a surface is unthemeable, cite the file and line where the custom rendering occurs.
- **Template must render.** For OWN-RENDERER and TERMINAL apps, you must verify the template produces valid output. Don't ship a template you haven't tested.
- **No source patches.** If you find a gap that requires source changes, document it as a P2 task in PATCHES.md. Do not patch here.
- **Be honest about coverage.** Do not mark surfaces as THEMED without verification. When in doubt, mark GAP and let QA_CHECKLIST decide.

---

## 6. Structured report (end of agent response — for QUEEN)

```
THEME_VERIFY REPORT
app_id: <app_id>
category: <category>
theme_delivery: gtk4-css.tmpl | gtk3-css.tmpl | qt5ct | iris-tera-template | userChrome.css | ansi-derivation | N/A
surfaces_themed: <N>
surfaces_gap: <N> (list them)
surfaces_na: <N>
template_created: yes | no | already-exists
template_path: lang/rice/components/11-theme-engine/impl/templates/<app_id>/<config>.tmpl (or N/A for TERMINAL: lang/rice/components/06-terminal/templates/<app_id>/<config>.tmpl)
template_render_verified: PASS | FAIL | N/A
dark_mode_wire_found: yes | no | N/A
reload_mechanism: <description or N/A>
p2_followup_items: <list of gaps that require source patching in P2, or "none">
patches_log: apps/docs/integration/<app_id>/PATCHES.md
```
