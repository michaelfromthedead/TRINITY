# PHASE 6 ARCHITECTURE: Localization and Polish

## Phase Overview

Phase 6 completes the tooling infrastructure with localization support, finalizing integration points, and polishing existing systems.

## Components

### 1. Localization System (engine/tooling/localization/)

**Purpose**: Complete localization pipeline from extraction to preview

**Architecture**:
```
LocalizationSystem
    |
    +-- TextExtractionTool
    |       +-- CodeExtractor
    |       |       +-- Python patterns: _(), localize(), tr(), ngettext()
    |       |       +-- C++ patterns: TR(), LOCTEXT()
    |       |       +-- JavaScript patterns: t(), i18n()
    |       +-- AssetExtractor
    |       |       +-- JSON key detection (text, label, title, etc.)
    |       |       +-- XML tag detection
    |       +-- ExtractionSource (9 types)
    |       +-- ExtractedString (text, location, key, context)
    |       +-- Deduplication
    |       +-- Export to JSON/CSV
    |
    +-- StringTableManager
    |       +-- StringTable (keyed collection)
    |       +-- StringEntry (translations dict, plurals)
    |       +-- StringContext (description, screenshot, max_length)
    |       +-- PluralRule (language-specific)
    |       |       +-- English: ONE/OTHER
    |       |       +-- Slavic: ONE/FEW/MANY
    |       |       +-- Arabic: ZERO/ONE/TWO/FEW/MANY/OTHER
    |       +-- Category organization
    |       +-- Text search
    |
    +-- TranslationMemoryManager
    |       +-- TranslationMemory (per language pair)
    |       +-- TMEntry (quality score, usage count, approved)
    |       +-- TMMatch (similarity, type: EXACT/FUZZY/CONTEXT)
    |       +-- Fuzzy matching algorithm:
    |       |       +-- Jaccard similarity (70% weight)
    |       |       +-- Length ratio (30% weight)
    |       |       +-- Context boost (+10%)
    |       +-- Indexed lookups (source, context)
    |
    +-- LocalizationWorkflow
    |       +-- WorkflowStep (EXTRACT, TRANSLATE, IMPORT, VALIDATE)
    |       +-- WorkflowState (PENDING, IN_PROGRESS, COMPLETED, FAILED)
    |       +-- TranslationTask (key, source, target, assignment)
    |       +-- Built-in validators:
    |       |       +-- check_empty (error)
    |       |       +-- check_placeholders (error)
    |       |       +-- check_untranslated (warning)
    |       |       +-- check_length (warning)
    |       +-- External tool export format
    |
    +-- LocalizationDashboard
    |       +-- LanguageProgress (total, translated, approved, words)
    |       +-- TranslationStats (completion, review counts)
    |       +-- MissingString (key, category, word count)
    |       +-- Export formats (text, JSON, CSV)
    |       +-- Sorting (completion, name, translated, words)
    |
    +-- LocalizationPreview
            +-- PreviewMode:
            |       +-- NORMAL: Actual translations
            |       +-- PSEUDO_LOC: Accented + expanded
            |       +-- KEYS_ONLY: Show [key.name]
            |       +-- MISSING_ONLY: Highlight untranslated
            |       +-- LONG_TEXT: Double length
            +-- PseudoLocalizer
            |       +-- Character replacement
            |       +-- Text expansion (default 1.3x)
            |       +-- Bracket wrapping [! text !]
            |       +-- RTL simulation
            +-- LanguageSwitcher
                    +-- Available languages
                    +-- Cycle next/previous
                    +-- Native language names
```

**Plural Rules**:
| Language | Rule Pattern |
|----------|-------------|
| English | n == 1 ? ONE : OTHER |
| Russian | n%10==1 && n%100!=11 ? ONE : n%10>=2&&n%10<=4&&(n%100<10 || n%100>=20) ? FEW : MANY |
| Arabic | n==0 ? ZERO : n==1 ? ONE : n==2 ? TWO : n%100>=3&&n%100<=10 ? FEW : n%100>=11 ? MANY : OTHER |

**Extraction Patterns**:
```python
# Python
_("text")
localize("text")
tr("text")
ngettext("singular", "plural")

# C++
TR("text")
LOCTEXT("context", "text")

# JavaScript/TypeScript
t("text")
i18n("text")
```

## Data Flow

### Extraction Pipeline
```
Source Files
    -> CodeExtractor.extract() / AssetExtractor.extract()
    -> ExtractedString collection
    -> Deduplication
    -> StringTable.add_string() for each
    -> Export to JSON/CSV for translators
```

### Translation Workflow
```
1. EXTRACT
    -> TextExtractionTool.extract_from_project()
    -> New strings added to StringTable

2. TRANSLATE
    -> Export for external translation
    -> TranslationTask created per string
    -> Translators work on tasks
    -> TM suggestions provided

3. IMPORT
    -> Import translated files
    -> TranslationTask.complete()
    -> TM updated with new translations

4. VALIDATE
    -> LocalizationWorkflow.validate_all()
    -> Validators run on each translation
    -> Errors/warnings reported
```

### Preview Flow
```
Runtime text request
    -> LocalizationPreview.get_text(key)
    -> Switch on PreviewMode:
        NORMAL: StringTableManager.get_text()
        PSEUDO_LOC: PseudoLocalizer.transform()
        KEYS_ONLY: return f"[{key}]"
        MISSING_ONLY: highlight if missing
        LONG_TEXT: return text * 2
```

## Integration Points

### Editor Integration
- String extraction from assets
- Translation workflow UI
- Preview mode toggle
- Dashboard panel

### Build Integration
- Validate strings during build
- Package per-language data
- Optimize string tables

### Runtime Integration
- String table loading
- Language switching
- Plural form selection
- Placeholder substitution

## Thread Safety

| Component | Strategy |
|-----------|----------|
| StringTableManager | Read-heavy, lock on write |
| TranslationMemory | Lock on update |
| LocalizationWorkflow | Per-task locking |
| LocalizationPreview | Thread-safe reads |

## Configuration

### Extraction Configuration
```python
ExtractionConfig(
    include_patterns=["*.py", "*.cpp", "*.ts"],
    exclude_patterns=["**/test/**", "**/generated/**"],
    json_keys=["text", "label", "title", "message"],
    dedup_strategy="first_occurrence",
)
```

### Workflow Configuration
```python
WorkflowConfig(
    auto_extract=True,
    validation_level="strict",
    tm_threshold=0.7,
    target_languages=["fr", "de", "ja", "zh-CN"],
)
```

### Preview Configuration
```python
PreviewConfig(
    default_mode=PreviewMode.NORMAL,
    pseudo_expansion=1.3,
    pseudo_brackets=True,
    rtl_simulation=False,
)
```

## Testing Strategy

### Unit Tests
- Pattern extraction correctness
- Plural rule evaluation
- Fuzzy matching accuracy
- Validator detection

### Integration Tests
- Full extraction pipeline
- Translation workflow cycle
- Preview mode switching

### Localization Tests
- All languages render
- RTL layout works
- Placeholder substitution
- Character encoding

---

## 2. Polish and Integration

### 2.1 Cross-Cutting Concerns

**Error Handling Audit**:
- [ ] All modules handle exceptions gracefully
- [ ] Error messages are user-friendly
- [ ] Recovery paths exist where possible

**Logging Consistency**:
- [ ] All modules use LogSystem
- [ ] Log levels appropriate
- [ ] Structured data where useful

**Thread Safety Review**:
- [ ] All singletons use RLock
- [ ] No race conditions
- [ ] Deadlock-free

### 2.2 Documentation

**API Documentation**:
- [ ] All public classes documented
- [ ] All public methods have docstrings
- [ ] Usage examples provided

**Architecture Documentation**:
- [ ] Component diagrams
- [ ] Data flow diagrams
- [ ] Integration guides

### 2.3 Performance Optimization

**Hot Path Analysis**:
- [ ] Profile critical paths
- [ ] Optimize bottlenecks
- [ ] Cache where beneficial

**Memory Optimization**:
- [ ] __slots__ on hot classes
- [ ] Weakrefs for long-lived objects
- [ ] Pool objects where appropriate

### 2.4 Test Coverage

**Coverage Goals**:
- [ ] 80% line coverage on core modules
- [ ] 100% coverage on critical paths
- [ ] Integration tests for all workflows
