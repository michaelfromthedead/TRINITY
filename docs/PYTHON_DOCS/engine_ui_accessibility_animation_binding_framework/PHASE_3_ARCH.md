# PHASE 3 ARCHITECTURE: Binding Module

## Scope

4 files, ~3,526 lines in `engine/ui/binding/`

| File | Lines | Purpose |
|------|-------|---------|
| binding.py | 1281 | MVVM data binding core |
| validation.py | 850 | Input validation system |
| converter.py | 752 | Value converter system |
| observable.py | 643 | Observable collections |

---

## Component Architecture

### Binding System (binding.py)

```
Binding
    |
    +-- source: Any (data model)
    +-- source_path: PropertyPath
    +-- target: Widget
    +-- target_property: str
    +-- mode: BindingMode
    |       +-- ONE_WAY (source → target)
    |       +-- TWO_WAY (source ↔ target)
    |       +-- ONE_TIME (source → target, once)
    |       +-- ONE_WAY_TO_SOURCE (target → source)
    |
    +-- converter: IConverter (optional)
    +-- validator: IValidator (optional)

MultiBinding
    +-- sources: List[BindingSource]
    +-- converter: MultiValueConverter
    +-- target: Widget
    +-- target_property: str

PropertyPath
    +-- segments: List[PathSegment]
    |       +-- name: str (property name)
    |       +-- is_indexer: bool
    |       +-- index: int | str (for indexers)
    |
    +-- parse("items[0].name") → [items, [0], name]

BindingContext
    +-- parent: BindingContext (optional)
    +-- data: Any
    +-- Lookup: child → parent chain

BindingGroup
    +-- bindings: List[Binding]
    +-- activate() / deactivate()
    +-- dispose()
```

**Property Path Examples**:
- `name` → single property
- `user.email` → nested property
- `items[0]` → list index
- `users["admin"].role` → dict key
- `orders[0].items[2].price` → deep nested

---

### Validation System (validation.py)

```
IValidator (interface)
    +-- validate(value, context) → ValidationResult

IAsyncValidator (interface)
    +-- validate_async(value, context) → Awaitable[ValidationResult]

ValidationResult
    +-- is_valid: bool
    +-- errors: List[str]

ValidationContext
    +-- property_name: str
    +-- parent_value: Any
    +-- custom_data: dict

Built-in Validators:
    +-- RequiredValidator
    +-- RangeValidator (min, max)
    +-- RegexValidator (pattern)
    +-- EmailValidator
    +-- UrlValidator
    +-- LengthValidator (min_length, max_length)

CompositeValidator
    +-- validators: List[IValidator]
    +-- mode: AND | OR

ValidationTrigger (enum)
    +-- ON_CHANGE: Validate on each keystroke
    +-- ON_BLUR: Validate when field loses focus
    +-- ON_SUBMIT: Validate only on form submit
```

---

### Converter System (converter.py)

```
IConverter (interface)
    +-- convert(value, target_type, parameter) → Any
    +-- convert_back(value, target_type, parameter) → Any

Built-in Converters:
    +-- BoolToVisibilityConverter
    |       True → Visibility.VISIBLE
    |       False → Visibility.COLLAPSED
    |
    +-- NumberFormatConverter
    |       1234.56 → "1,234.56"
    |       parse back with locale
    |
    +-- ColorConverter
    |       "#FF0000" → Color(255, 0, 0)
    |       Color → "#FF0000"
    |
    +-- DateFormatConverter
    |       datetime → "2026-05-23"
    |       parse back with format

ChainedConverter
    +-- converters: List[IConverter]
    +-- convert: apply in order
    +-- convert_back: apply in reverse

LambdaConverter
    +-- convert_func: Callable
    +-- convert_back_func: Callable (optional)

CachedConverter
    +-- inner: IConverter
    +-- cache: WeakKeyDictionary

MultiValueConverter (interface)
    +-- convert(values: List[Any], ...) → Any
    +-- convert_back(value, ...) → List[Any]
```

---

### Observable Collections (observable.py)

```
ObservableList[T]
    +-- _items: List[T]
    +-- _listeners: List[WeakRef[Callable]]
    |
    +-- add(item) → notify ADD
    +-- remove(item) → notify REMOVE
    +-- replace(index, item) → notify REPLACE
    +-- move(old_index, new_index) → notify MOVE
    +-- clear() → notify RESET
    |
    +-- begin_batch() / end_batch() → single notification

ObservableDict[K, V]
    +-- _items: Dict[K, V]
    +-- Same notification pattern

CollectionChangeEvent
    +-- action: ADD | REMOVE | REPLACE | MOVE | RESET
    +-- items: List[T]
    +-- old_index: int (optional)
    +-- new_index: int (optional)

VirtualizedListView
    +-- source: ObservableList
    +-- viewport_height: float
    +-- item_height: float
    +-- scroll_offset: float
    |
    +-- _active_widgets: Dict[int, Widget]
    +-- _recycled_widgets: List[Widget]
    |
    +-- _update_visible_range()
    +-- _create_widget(index)
    +-- _recycle_widget(index)
```

**Virtualization Formula**:
```python
first_visible = scroll_offset // item_height
visible_count = viewport_height // item_height + 2  # buffer
last_visible = min(first_visible + visible_count, len(source))
```

---

## Data Flow

### One-Way Binding

```
Source Model
    |
    v
PropertyPath.get_value()
    |
    v
Converter.convert()
    |
    v
Target Widget Property
```

### Two-Way Binding

```
Source Model ←────────────────┐
    |                          |
    v                          |
PropertyPath.get_value()       |
    |                          |
    v                          |
Converter.convert()            |
    |                          |
    v                          |
Target Widget Property         |
    |                          |
    v                          |
User Input                     |
    |                          |
    v                          |
Validator.validate()           |
    |                          |
    v (if valid)               |
Converter.convert_back() ──────┘
    |
    v
PropertyPath.set_value()
    |
    v
Source Model
```

### Observable Collection

```
ObservableList.add(item)
    |
    v
CollectionChangeEvent(ADD, [item])
    |
    v
Notify all listeners
    |
    v
VirtualizedListView._on_collection_changed()
    |
    v
If index in visible range:
    _create_widget(index)
```

---

## Integration Points

| From | To | Purpose |
|------|----|---------| 
| Widget | Binding | Property change notification |
| Binding | DataTrigger | Value change triggers animation |
| Validator | Widget | Error state display |
| ObservableList | VirtualizedListView | Collection updates |
| Converter | Localization | Formatted values |

---

## Design Decisions

### D1: Property Path with Indexers

**Decision**: Support dot notation with array/dict indexers.

**Rationale**: Real data models are nested. Flattening models for binding is impractical.

**Syntax**: `users[0].addresses["home"].city`

### D2: Weak Reference Listeners

**Decision**: Use weak references for collection change listeners.

**Rationale**: Prevents memory leaks when widgets are disposed without explicit unbinding.

### D3: Validation Triggers

**Decision**: Three trigger modes (ON_CHANGE, ON_BLUR, ON_SUBMIT).

**Rationale**: Different UX patterns. ON_CHANGE is aggressive, ON_SUBMIT is lazy.

### D4: Async Validation

**Decision**: Support async validators for server-side checks.

**Rationale**: Username availability, email verification require server round-trip.

**Cancellation**: Previous async validation cancelled when new validation starts.

### D5: Chained Converters

**Decision**: Converters can be composed in chains.

**Rationale**: Reuse simple converters. BoolToVisibility + VisibilityToOpacity = BoolToOpacity.

### D6: Widget Recycling

**Decision**: VirtualizedListView recycles widget instances.

**Rationale**: Creating/destroying widgets for 10K items is slow. Recycling reuses widget instances with new data.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Deep property path performance | Slow updates | Cache resolved paths |
| Circular binding | Infinite loop | Track update source, skip if self |
| Async validation race | Wrong result shown | Cancel previous, show latest |
| Virtualization edge cases | Missing/duplicate items | Test scroll boundaries |
| Weak ref GC timing | Stale listeners | Manual cleanup on dispose |
