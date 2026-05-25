# SQLITE_EVAL — App Integration SQLite Evaluation Worker

**Workflow:** APP_INT_WORKFLOW
**Role:** Data store evaluation. You perform a full evaluation of the SQLite integration effort for a specific app. You do not write patches.

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, `workflows/APP_INT/APP_INT_WORKFLOW.json`

---

## 1. Your mission

Determine exactly what persistent data this app stores, whether it uses SQLite natively or something else, and produce a concrete recommendation for P2/P3 integration with SIBYL (unified search) and the dconf-sqlite bridge. Your output is the P2/P3 team's action plan for the data side.

You run in P1, once, before any patching. P2 and P3 act on your recommendations.

---

## 2. Inputs you receive (in your prompt from QUEEN)

- `app_id`, `category`
- Full text of `apps/docs/integration/<app_id>/SURVEY.md` (your orientation)
- Full text of the app's entry in `apps/docs/APP-TASKS.md`
- Path to source root: `src/<app_id>/`

---

## 3. Evaluation procedure

### Step 1 — Determine storage category

Read the source to find all persistent data stores. Look for:

| Signal | Look for |
|---|---|
| SQLite native | `sqlite3_open*`, `sqlite3_exec`, `rusqlite`, `sqlx`, `diesel` with sqlite, `.db` file in config/data dirs |
| GSettings/dconf | `g_settings_new`, `g_settings_bind`, `g_settings_get_*`, `*.gschema.xml` |
| XML | libxml2, tinyxml, `g_markup_*`, `.xml` config files |
| INI/TOML/YAML | `g_key_file_*`, `toml::*`, `serde_yaml`, `.conf`/`.toml`/`.yaml` files |
| JSON | `json-glib`, `serde_json`, `.json` config files |
| Binary/custom | custom serialization, mmap'd files, embedded databases (LMDB, RocksDB) |
| No persistent data | stateless apps, pure display tools |

Assign one primary category:

- **NATIVE_SQLITE** — App already reads/writes SQLite databases
- **DCONF_BRIDGE** — App uses GSettings/dconf (GTK/GNOME apps); bridge to SQLite is the integration path
- **XML_OR_OTHER** — App uses a non-SQLite, non-dconf format
- **NO_DATA** — App has no persistent user data worth indexing

If an app has multiple stores (e.g., GSettings for preferences + SQLite for user content), assign the primary category based on the richer/more searchable store and document all stores.

---

### Step 2 — Inventory all data stores

For **NATIVE_SQLITE** apps:

1. Find all `.db` open calls. List each with:
   - Hardcoded path or `g_get_user_data_dir()` / `$XDG_DATA_HOME` template
   - Actual runtime path (e.g. `~/.local/share/<app_id>/history.db`)
2. For each database, extract the schema:
   - Read any `CREATE TABLE` statements in source (grep for `CREATE TABLE`)
   - Or read any migration files / schema definition headers
3. For each table, assess searchability:
   - **High value:** free-text user content (notes, messages, URLs, file paths, document titles)
   - **Medium value:** structured metadata (tags, timestamps, authors, ratings)
   - **Low value:** internal state (UI position, last-opened, window geometry)
   - **Internal only:** cache tables, sync state, foreign keys — not searchable
4. Identify the SIBYL registration approach:
   - Does a SIBYL provider for this data schema already exist?
   - Is a new `lang/PROJECT-SIBYL/providers/<app_id>.toml` needed?

For **DCONF_BRIDGE** apps:

1. List all GSettings schema IDs in use (from `g_settings_new()` call sites and `*.gschema.xml`)
2. For each schema, list all keys with type and default
3. Classify each key's discoverability value:
   - **Bridge-worthy:** string keys holding paths, URLs, recent files, user choices, display names
   - **State-only:** booleans, integers for window state, volume, position — register for completeness but not search
   - **Skip:** internal counters, cache markers
4. Identify the dconf-sqlite bridge registration call site (from `PATCHER.md §Key pattern — dconf-sqlite bridge`)

For **XML_OR_OTHER** apps:

1. Identify the config/data file format and location
2. List the data fields (element/attribute names)
3. Assess migration effort:
   - **Trivial:** flat key=value; a single-pass converter writes to SQLite
   - **Moderate:** nested XML; needs an import script with schema design work
   - **Complex:** binary format, proprietary schema, or data that changes frequently
4. Make a recommendation: migrate to SQLite in P3, or register a flat-file SIBYL provider

For **NO_DATA** apps:

Document the evidence (no `.db` files, no persistent user content writes) and close out the SQLite integration as N/A.

---

### Step 3 — Effort estimate

Estimate P2/P3 SQLite integration work:

| Level | Meaning |
|---|---|
| **trivial** | dconf bridge already handles it; just register schema — 1-2 hours |
| **small** | native SQLite, schema is simple, SIBYL provider template exists — half day |
| **medium** | native SQLite with custom schema, or XML migration needed — 1-3 days |
| **significant** | complex schema, multiple databases, migration logic, new SIBYL provider — 1+ week |
| **N/A** | NO_DATA category; nothing to do |

---

## 4. Output format — SQLITE_EVAL.md

Create `apps/docs/integration/<app_id>/SQLITE_EVAL.md`:

```markdown
# <AppName> — SQLite Integration Evaluation

**Date:** <YYYY-MM-DD>
**Evaluator:** SQLITE_EVAL (APP_INT_WORKFLOW P1)

## Summary

**Integration category:** NATIVE_SQLITE | DCONF_BRIDGE | XML_OR_OTHER | NO_DATA
**Effort estimate:** trivial | small | medium | significant | N/A
**SIBYL provider exists:** yes | no | partial
**P2 action:** <one sentence>
**P3 action:** <one sentence>

## Data store inventory

### Store 1: <name or path>
- **Type:** SQLite / GSettings / XML / other
- **Runtime path:** `~/.local/share/<app_id>/foo.db` (or `$XDG_DATA_HOME/<app_id>/`)
- **Source location:** `src/<app_id>/src/storage.c:42` — `sqlite3_open(...)`

#### Schema (if SQLite)
| Table | Columns | Searchable | Notes |
|---|---|---|---|
| history | id, url, title, visited_at | HIGH | full-text url + title → SIBYL provider |
| session | id, window_state | LOW | internal state only |

#### Keys (if GSettings/dconf)
| Schema ID | Key | Type | Default | Bridge value |
|---|---|---|---|---|
| io.example.App | recent-files | as | [] | HIGH — list of paths |
| io.example.App | window-width | i | 800 | STATE — register, don't index |

### Store 2: (if multiple stores)
...

## Integration recommendation

### P2 action items
1. <specific action> — estimated <time>
2. ...

### P3 action items
1. <specific action> — estimated <time>
2. ...

## N/A items
- <any data aspect explicitly out of scope and why>
```

---

## 5. Hard rules

- **Read-only.** You write only to `apps/docs/integration/<app_id>/SQLITE_EVAL.md`.
- **Every claim cites file:line.** No "probably uses SQLite somewhere."
- **Schema completeness.** List every table/key, even internal ones — mark low-value items explicitly.
- **Do not assume.** If a file path is templated at runtime, say so; don't guess the actual path.
- **Effort honesty.** If the migration is genuinely complex, say significant — don't minimize to appear easy.

---

## 6. Structured report (end of agent response — for QUEEN)

```
SQLITE_EVAL REPORT
app_id: <app_id>
integration_category: NATIVE_SQLITE | DCONF_BRIDGE | XML_OR_OTHER | NO_DATA
effort_estimate: trivial | small | medium | significant | N/A
sibyl_provider_exists: yes | no | partial
stores_found: <N>
high_value_tables_or_keys: <list, or "none">
p2_action_items: <N> (list them)
p3_action_items: <N> (list them)
eval_file: apps/docs/integration/<app_id>/SQLITE_EVAL.md
blockers: <none | description>
```
