# PHASE 3 TODO: Binding Module

## Summary

Verify and test the 4 binding files (~3,526 lines) for correct MVVM implementation, validation, converters, and observables.

---

## T1: Property Path Navigation

**File**: `engine/ui/binding/binding.py`

### T1.1: Test Simple Property Access
- [ ] Path "name" gets `obj.name`
- [ ] Path "name" sets `obj.name = value`
- [ ] Missing property raises AttributeError

**Acceptance**: Simple paths work correctly.

### T1.2: Test Nested Property Access
- [ ] Path "user.email" navigates two levels
- [ ] Path "a.b.c.d" navigates four levels
- [ ] Any missing segment raises error

**Acceptance**: Nested paths navigate correctly.

### T1.3: Test List Indexer Access
- [ ] Path "items[0]" gets first element
- [ ] Path "items[2]" gets third element
- [ ] Path "items[-1]" gets last element
- [ ] Out of range raises IndexError

**Acceptance**: List indexers work correctly.

### T1.4: Test Dict Indexer Access
- [ ] Path "users['admin']" gets dict value
- [ ] Path 'settings["theme"]' gets dict value
- [ ] Missing key raises KeyError

**Acceptance**: Dict indexers work correctly.

### T1.5: Test Mixed Path Access
- [ ] Path "orders[0].items[2].price" navigates mixed structure
- [ ] Path "config['db'].hosts[0].port" navigates mixed structure

**Acceptance**: Complex paths work correctly.

---

## T2: Binding Modes

**File**: `engine/ui/binding/binding.py`

### T2.1: Test ONE_WAY Binding
- [ ] Source change updates target
- [ ] Target change does NOT update source
- [ ] Initial value propagates to target

**Acceptance**: One-way flow only.

### T2.2: Test TWO_WAY Binding
- [ ] Source change updates target
- [ ] Target change updates source
- [ ] No infinite loop on change

**Acceptance**: Bidirectional flow.

### T2.3: Test ONE_TIME Binding
- [ ] Initial value propagates to target
- [ ] Subsequent source changes ignored
- [ ] Target changes ignored

**Acceptance**: Single initial propagation.

### T2.4: Test ONE_WAY_TO_SOURCE Binding
- [ ] Target change updates source
- [ ] Source change does NOT update target
- [ ] Initial value NOT propagated

**Acceptance**: Reverse one-way flow.

---

## T3: MultiBinding

**File**: `engine/ui/binding/binding.py`

### T3.1: Test Multiple Sources
- [ ] Two sources combine via converter
- [ ] Three sources combine via converter
- [ ] Any source change triggers update

**Acceptance**: Multiple sources work.

### T3.2: Test MultiValueConverter
- [ ] convert([a, b]) → combined value
- [ ] convert_back(combined) → [a, b]
- [ ] Partial convert_back (return UnsetValue for unchanged)

**Acceptance**: Multi-value conversion works.

---

## T4: BindingContext

**File**: `engine/ui/binding/binding.py`

### T4.1: Test Context Hierarchy
- [ ] Child context inherits parent data
- [ ] Child can override parent properties
- [ ] Lookup walks up the chain

**Acceptance**: Context inheritance works.

### T4.2: Test Context Change
- [ ] Changing context updates all bindings
- [ ] Old bindings cleaned up
- [ ] New bindings activated

**Acceptance**: Context switching works.

---

## T5: Validation System

**File**: `engine/ui/binding/validation.py`

### T5.1: Test RequiredValidator
- [ ] Empty string → invalid
- [ ] None → invalid
- [ ] "value" → valid
- [ ] Whitespace-only → invalid (if configured)

**Acceptance**: Required validation works.

### T5.2: Test RangeValidator
- [ ] Below min → invalid with message
- [ ] Above max → invalid with message
- [ ] Within range → valid
- [ ] Boundary values → valid

**Acceptance**: Range validation works.

### T5.3: Test RegexValidator
- [ ] Matching pattern → valid
- [ ] Non-matching → invalid
- [ ] Custom error message shown

**Acceptance**: Regex validation works.

### T5.4: Test EmailValidator
- [ ] "user@example.com" → valid
- [ ] "user@domain" → invalid (no TLD)
- [ ] "user" → invalid (no @)
- [ ] "@example.com" → invalid (no local part)

**Acceptance**: Email validation works.

### T5.5: Test UrlValidator
- [ ] "https://example.com" → valid
- [ ] "http://localhost:8080" → valid
- [ ] "example.com" → invalid (no scheme)
- [ ] "ftp://files.com" → valid or invalid (configurable)

**Acceptance**: URL validation works.

### T5.6: Test CompositeValidator AND
- [ ] All pass → valid
- [ ] Any fail → invalid
- [ ] All errors collected

**Acceptance**: AND composition works.

### T5.7: Test CompositeValidator OR
- [ ] Any pass → valid
- [ ] All fail → invalid

**Acceptance**: OR composition works.

### T5.8: Test Async Validation
- [ ] Async validator returns result
- [ ] Multiple async validators run in parallel
- [ ] Cancellation works when new validation starts

**Acceptance**: Async validation works.

### T5.9: Test Validation Triggers
- [ ] ON_CHANGE validates on each input
- [ ] ON_BLUR validates when focus lost
- [ ] ON_SUBMIT validates only on submit

**Acceptance**: Trigger modes work correctly.

---

## T6: Converter System

**File**: `engine/ui/binding/converter.py`

### T6.1: Test BoolToVisibilityConverter
- [ ] True → VISIBLE
- [ ] False → COLLAPSED
- [ ] convert_back: VISIBLE → True
- [ ] convert_back: COLLAPSED → False

**Acceptance**: Bool to visibility works.

### T6.2: Test NumberFormatConverter
- [ ] 1234.56 → "1,234.56" (with locale)
- [ ] convert_back: "1,234.56" → 1234.56
- [ ] Invalid string → error or default

**Acceptance**: Number formatting works.

### T6.3: Test ColorConverter
- [ ] "#FF0000" → Color(255, 0, 0)
- [ ] "rgb(255, 0, 0)" → Color(255, 0, 0)
- [ ] Color → "#FF0000"

**Acceptance**: Color conversion works.

### T6.4: Test ChainedConverter
- [ ] Two converters chain: A → B → C
- [ ] convert_back reverses: C → B → A
- [ ] Order matters

**Acceptance**: Chained conversion works.

### T6.5: Test LambdaConverter
- [ ] Custom convert function works
- [ ] Custom convert_back function works
- [ ] One-way (no convert_back) works

**Acceptance**: Lambda conversion works.

### T6.6: Test CachedConverter
- [ ] Same input returns cached output
- [ ] Cache uses weak keys (no memory leak)
- [ ] Cache cleared on converter reset

**Acceptance**: Caching works correctly.

---

## T7: Observable Collections

**File**: `engine/ui/binding/observable.py`

### T7.1: Test ObservableList Add
- [ ] add(item) notifies listeners
- [ ] Event contains ADD action
- [ ] Event contains added item
- [ ] Event contains new index

**Acceptance**: Add notification works.

### T7.2: Test ObservableList Remove
- [ ] remove(item) notifies listeners
- [ ] Event contains REMOVE action
- [ ] Event contains removed item
- [ ] Event contains old index

**Acceptance**: Remove notification works.

### T7.3: Test ObservableList Replace
- [ ] replace(index, item) notifies
- [ ] Event contains REPLACE action
- [ ] Event contains old and new item
- [ ] Event contains index

**Acceptance**: Replace notification works.

### T7.4: Test ObservableList Move
- [ ] move(old, new) notifies
- [ ] Event contains MOVE action
- [ ] Event contains old and new indices

**Acceptance**: Move notification works.

### T7.5: Test ObservableList Clear
- [ ] clear() notifies with RESET
- [ ] Single notification for entire clear
- [ ] List is empty after clear

**Acceptance**: Clear notification works.

### T7.6: Test Batch Operations
- [ ] begin_batch() suppresses notifications
- [ ] end_batch() fires single notification
- [ ] Notification contains all changes

**Acceptance**: Batching works correctly.

### T7.7: Test Weak References
- [ ] Listener garbage collected when dereferenced
- [ ] No notification to dead listener
- [ ] No memory leak

**Acceptance**: Weak references work.

---

## T8: Virtualized List View

**File**: `engine/ui/binding/observable.py`

### T8.1: Test Visible Range Calculation
- [ ] First visible = scroll_offset / item_height
- [ ] Last visible = first + viewport_height / item_height + buffer
- [ ] Clamped to list bounds

**Acceptance**: Range calculation correct.

### T8.2: Test Widget Creation
- [ ] Widget created for visible index
- [ ] Widget positioned correctly
- [ ] Widget bound to correct data

**Acceptance**: Widget creation works.

### T8.3: Test Widget Recycling
- [ ] Scrolled-out widget moved to recycle pool
- [ ] Recycled widget reused for new index
- [ ] Widget rebound to new data

**Acceptance**: Recycling works.

### T8.4: Test Scroll Performance
- [ ] Rapid scrolling doesn't create excess widgets
- [ ] Buffer prevents visible gaps
- [ ] 10K items scrolls smoothly

**Acceptance**: Virtualization performant.

### T8.5: Test Collection Updates
- [ ] ADD in visible range creates widget
- [ ] REMOVE in visible range removes widget
- [ ] RESET recreates visible widgets

**Acceptance**: Collection changes handled.

---

## Completion Criteria

All tasks T1-T8 marked complete with tests passing.
