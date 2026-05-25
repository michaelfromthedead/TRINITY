# FlowForge Part 2: Foundation Integration

**Status:** Future work (after Projects 1-4 complete)

---

## Overview

Part 2 replaces FlowForge's direct Trinity access with a Foundation adapter. This gives FlowForge access to everything Trinity exposes, plus Foundation's runtime services.

```
BEFORE (Part 1):                      AFTER (Part 2):

flowforge ──► trinity                 flowforge ──► foundation ──► trinity
                                                        │
                                                        ├── registry
                                                        ├── tracker  
                                                        ├── eventlog
                                                        └── mirror
```

---

## Why This Change

### Current Approach (Projects 1-4)

```python
# flowforge_backend/trinity_adapter.py
from trinity.metaclasses.component_meta import ComponentMeta

def build_node_schemas():
    """Direct Trinity access - works but limited."""
    for cls in ComponentMeta.all_components():
        # Only sees class definitions
        # No runtime state
        # No change tracking
        # No event history
```

### Improved Approach (Part 2)

```python
# flowforge_backend/foundation_adapter.py
from foundation import registry, tracker, eventlog, mirror

def build_node_schemas():
    """Foundation access - single source of truth."""
    for cls in registry.all_types():
        yield {
            "class": cls,
            "instances": list(registry.instances(cls)),
            "dirty_fields": tracker.get_dirty(cls),
            "history": eventlog.by_entity(cls),
        }
```

---

## What Foundation Provides

| System | What FlowForge Gains |
|--------|----------------------|
| `registry` | All types, all instances, decorator metadata |
| `tracker` | Dirty flags, undo/redo state, change subscriptions |
| `eventlog` | Full operation history with causal chains |
| `mirror` | Deep runtime introspection |
| `bridge` | ShellLang world sync (if needed) |

---

## New Features Enabled

### 1. Live Instance View

See actual runtime objects, not just class definitions:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Player (class)  │     │ player_1        │     │ player_2        │
│ ═══════════════ │     │ ═══════════════ │     │ ═══════════════ │
│ health: int     │     │ health: 100     │     │ health: 45      │
│ position: Vec3  │     │ position: (0,0) │     │ position: (5,3) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
      CLASS                 INSTANCE               INSTANCE
```

### 2. Dirty Field Highlighting

Tracker knows which fields changed since last save/checkpoint:

```
┌─────────────────┐
│ player_1        │
│ ═══════════════ │
│ health: 100     │  ← normal
│ position: (5,3) │  ← YELLOW (dirty, was 0,0)
└─────────────────┘
```

### 3. Event Timeline

Visualize the eventlog as a timeline panel:

```
┌─────────────────────────────────────────────────────────────────┐
│ EVENT TIMELINE                                                  │
├─────────────────────────────────────────────────────────────────┤
│ tick 0   │ spawn_player      │ created player_1                │
│ tick 1   │ move_player       │ player_1.position: (0,0)→(1,0)  │
│ tick 2   │ move_player       │ player_1.position: (1,0)→(2,0)  │
│ tick 3   │ apply_damage      │ player_1.health: 100→85         │
│          │   └─ caused by    │ enemy_attack (enemy_1)          │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Causal Chain Visualization

Click a value to see why it has that value:

```
player_1.health = 85

WHY?
└── apply_damage (tick 3)
    └── caused by: enemy_attack
        └── caused by: enemy_ai_tick
            └── root cause: game_loop_tick
```

### 5. Undo/Redo Integration

Tracker's undo/redo exposed as visual controls:

```
┌────────────────────────────────────────┐
│  [◄ Undo]  [Redo ►]   tick 3 of 47    │
└────────────────────────────────────────┘
```

### 6. Change Subscriptions

FlowForge subscribes to Foundation changes for live updates:

```python
# foundation_adapter.py
def connect_live_updates(callback):
    tracker.on_change(lambda obj, field, old, new: 
        callback({
            "type": "field_changed",
            "entity": id(obj),
            "field": field,
            "old": old,
            "new": new,
        })
    )
```

Canvas updates automatically when game state changes.

---

## Implementation Plan

### Phase 2.1: Create Foundation Adapter

| Task | Description |
|------|-------------|
| 2.1.1 | Create `flowforge_backend/foundation_adapter.py` |
| 2.1.2 | Implement `get_all_types()` via registry |
| 2.1.3 | Implement `get_instances(cls)` via registry |
| 2.1.4 | Implement `get_dirty_fields(obj)` via tracker |
| 2.1.5 | Implement `get_history(entity_id)` via eventlog |
| 2.1.6 | Implement `subscribe_changes(callback)` via tracker |

### Phase 2.2: Replace Trinity Adapter

| Task | Description |
|------|-------------|
| 2.2.1 | Update `build_node_schemas()` to use foundation_adapter |
| 2.2.2 | Remove direct Trinity imports from flowforge_backend |
| 2.2.3 | Update IPC handlers to use new adapter |
| 2.2.4 | Test that existing functionality still works |

### Phase 2.3: Add Runtime Features

| Task | Description |
|------|-------------|
| 2.3.1 | Add instance view mode to frontend |
| 2.3.2 | Add dirty field highlighting |
| 2.3.3 | Add event timeline panel |
| 2.3.4 | Add causal chain inspector |
| 2.3.5 | Add undo/redo controls |
| 2.3.6 | Wire live change subscriptions |

---

## File Changes

### New Files

```
flowforge_backend/
└── foundation_adapter.py    # New adapter replacing trinity_adapter.py
```

### Modified Files

```
flowforge_backend/
├── trinity_adapter.py       # Deprecated, calls foundation_adapter
├── ipc/handlers.py          # Use foundation_adapter
└── __main__.py              # Initialize foundation connection
```

### Frontend Additions

```
apps/desktop/src/
├── components/
│   ├── InstancePanel.vue    # Show live instances
│   ├── EventTimeline.vue    # Visualize eventlog
│   └── CausalInspector.vue  # "Why does X = Y?"
└── stores/
    └── runtime.ts           # Live state from Foundation
```

---

## API Design

### foundation_adapter.py

```python
"""
FlowForge Foundation Adapter

Single point of contact between FlowForge and the Foundation runtime.
Replaces direct Trinity access with Foundation's unified view.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Type
from dataclasses import dataclass

@dataclass
class TypeInfo:
    """Information about a registered type."""
    cls: Type
    name: str
    fields: Dict[str, type]
    decorators: List[str]
    instance_count: int

@dataclass  
class InstanceInfo:
    """Information about a live instance."""
    obj: Any
    type_name: str
    entity_id: int
    field_values: Dict[str, Any]
    dirty_fields: List[str]

@dataclass
class ChangeEvent:
    """A change notification from the tracker."""
    entity_id: int
    field: str
    old_value: Any
    new_value: Any
    tick: Optional[int]

class FoundationAdapter:
    """
    Adapter providing FlowForge access to Foundation systems.
    
    This is the ONLY point where FlowForge touches the engine internals.
    All Trinity access goes through Foundation's registry/tracker/eventlog.
    """
    
    def __init__(self):
        self._change_subscribers: List[Callable[[ChangeEvent], None]] = []
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to Foundation systems. Returns True if successful."""
        try:
            from foundation import registry, tracker, eventlog, mirror
            self._registry = registry
            self._tracker = tracker
            self._eventlog = eventlog
            self._mirror = mirror
            self._connected = True
            
            # Subscribe to tracker changes
            tracker.on_change(self._on_change)
            
            return True
        except ImportError:
            return False
    
    # === Type Information ===
    
    def get_all_types(self) -> Iterator[TypeInfo]:
        """Get all registered types."""
        for cls in self._registry.all_types():
            yield TypeInfo(
                cls=cls,
                name=cls.__name__,
                fields=self._get_fields(cls),
                decorators=self._get_decorators(cls),
                instance_count=len(list(self._registry.instances(cls))),
            )
    
    def get_type(self, name: str) -> Optional[TypeInfo]:
        """Get a specific type by name."""
        cls = self._registry.get(name)
        if cls:
            return TypeInfo(
                cls=cls,
                name=name,
                fields=self._get_fields(cls),
                decorators=self._get_decorators(cls),
                instance_count=len(list(self._registry.instances(cls))),
            )
        return None
    
    # === Instance Information ===
    
    def get_instances(self, type_name: str) -> Iterator[InstanceInfo]:
        """Get all live instances of a type."""
        cls = self._registry.get(type_name)
        if not cls:
            return
        
        for obj in self._registry.instances(cls):
            yield InstanceInfo(
                obj=obj,
                type_name=type_name,
                entity_id=getattr(obj, 'id', id(obj)),
                field_values=self._get_field_values(obj),
                dirty_fields=self._tracker.get_dirty_fields(obj),
            )
    
    def get_instance(self, entity_id: int) -> Optional[InstanceInfo]:
        """Get a specific instance by entity ID."""
        # Use mirror for lookup
        obj = self._mirror.get_by_id(entity_id)
        if obj:
            return InstanceInfo(
                obj=obj,
                type_name=type(obj).__name__,
                entity_id=entity_id,
                field_values=self._get_field_values(obj),
                dirty_fields=self._tracker.get_dirty_fields(obj),
            )
        return None
    
    # === Change Tracking ===
    
    def get_dirty_fields(self, entity_id: int) -> List[str]:
        """Get list of dirty fields for an entity."""
        obj = self._mirror.get_by_id(entity_id)
        if obj:
            return self._tracker.get_dirty_fields(obj)
        return []
    
    def subscribe_changes(self, callback: Callable[[ChangeEvent], None]) -> None:
        """Subscribe to change notifications."""
        self._change_subscribers.append(callback)
    
    def unsubscribe_changes(self, callback: Callable[[ChangeEvent], None]) -> None:
        """Unsubscribe from change notifications."""
        self._change_subscribers.remove(callback)
    
    def _on_change(self, obj: Any, field: str, old: Any, new: Any) -> None:
        """Internal handler for tracker changes."""
        event = ChangeEvent(
            entity_id=getattr(obj, 'id', id(obj)),
            field=field,
            old_value=old,
            new_value=new,
            tick=self._eventlog.current_tick if hasattr(self._eventlog, 'current_tick') else None,
        )
        for callback in self._change_subscribers:
            callback(event)
    
    # === Event History ===
    
    def get_history(self, entity_id: int, limit: int = 100) -> List[dict]:
        """Get event history for an entity."""
        events = self._eventlog.by_entity(entity_id)
        return [
            {
                "tick": e.tick,
                "operation": e.operation,
                "changes": [
                    {"field": c.field, "old": c.old_value, "new": c.new_value}
                    for c in e.changes
                ],
                "cause": e.cause,
            }
            for e in events[:limit]
        ]
    
    def get_causal_chain(self, entity_id: int, field: str) -> List[dict]:
        """Get the causal chain explaining a field's current value."""
        return self._eventlog.trace_causation(entity_id, field)
    
    # === Undo/Redo ===
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._tracker.can_undo()
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._tracker.can_redo()
    
    def undo(self) -> bool:
        """Perform undo. Returns True if successful."""
        return self._tracker.undo()
    
    def redo(self) -> bool:
        """Perform redo. Returns True if successful."""
        return self._tracker.redo()
    
    # === Private Helpers ===
    
    def _get_fields(self, cls: Type) -> Dict[str, type]:
        """Extract field definitions from a class."""
        # Use __annotations__ and Trinity's field descriptors
        fields = {}
        for name, typ in getattr(cls, '__annotations__', {}).items():
            fields[name] = typ
        return fields
    
    def _get_decorators(self, cls: Type) -> List[str]:
        """Extract decorator names from a class."""
        decorators = []
        if hasattr(cls, '_component_name'):
            decorators.append('component')
        if hasattr(cls, '_system_queries'):
            decorators.append('system')
        if hasattr(cls, '_is_resource'):
            decorators.append('resource')
        if hasattr(cls, '_event_payload'):
            decorators.append('event')
        return decorators
    
    def _get_field_values(self, obj: Any) -> Dict[str, Any]:
        """Get current field values from an instance."""
        values = {}
        for name in getattr(type(obj), '__annotations__', {}):
            if hasattr(obj, name):
                values[name] = getattr(obj, name)
        return values


# Singleton instance
_adapter: Optional[FoundationAdapter] = None

def get_adapter() -> FoundationAdapter:
    """Get the global Foundation adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = FoundationAdapter()
    return _adapter
```

---

## Migration Path

### Backward Compatibility

During migration, `trinity_adapter.py` becomes a thin wrapper:

```python
# flowforge_backend/trinity_adapter.py (deprecated)
"""
DEPRECATED: Use foundation_adapter instead.

This module is kept for backward compatibility during migration.
All calls are forwarded to foundation_adapter.
"""

from .foundation_adapter import get_adapter

def build_node_schemas():
    """Deprecated. Use foundation_adapter.get_all_types() instead."""
    adapter = get_adapter()
    adapter.connect()
    
    schemas = {}
    for type_info in adapter.get_all_types():
        schemas[type_info.name] = {
            "fields": type_info.fields,
            "decorators": type_info.decorators,
        }
    return schemas
```

### Deprecation Timeline

1. **Part 2.1**: Add foundation_adapter alongside trinity_adapter
2. **Part 2.2**: Update all code to use foundation_adapter
3. **Part 2.3**: Mark trinity_adapter as deprecated
4. **Part 2.4**: Remove trinity_adapter

---

## Testing

### Unit Tests

```python
# tests/test_foundation_adapter.py

def test_connect_to_foundation():
    adapter = FoundationAdapter()
    assert adapter.connect() == True

def test_get_all_types():
    adapter = get_adapter()
    adapter.connect()
    types = list(adapter.get_all_types())
    assert len(types) > 0
    assert all(isinstance(t, TypeInfo) for t in types)

def test_change_subscription():
    adapter = get_adapter()
    adapter.connect()
    
    events = []
    adapter.subscribe_changes(events.append)
    
    # Trigger a change via Trinity
    player = Player(health=100)
    player.health = 50
    
    assert len(events) == 1
    assert events[0].field == "health"
    assert events[0].old_value == 100
    assert events[0].new_value == 50
```

### Integration Tests

```python
def test_live_updates_reach_frontend():
    """Test that changes in Trinity propagate to FlowForge UI."""
    # Start FlowForge with test game
    # Make change via game code
    # Verify FlowForge canvas updates
    pass

def test_undo_redo_via_flowforge():
    """Test undo/redo through FlowForge UI."""
    # Make changes via FlowForge
    # Click undo
    # Verify game state reverted
    pass
```

---

## Summary

Part 2 is a refactoring that:

1. **Removes** direct Trinity access from FlowForge
2. **Adds** Foundation adapter as single integration point
3. **Enables** live instance view, dirty highlighting, event timeline, causal inspection, undo/redo

The changeset is small but the capability gain is significant. FlowForge goes from "static class viewer" to "live runtime debugger."

---

## Prerequisites

Complete Projects 1-4 first:
- [ ] Project 1: Native shell working
- [ ] Project 2: AST parser working  
- [ ] Project 3: View mode working
- [ ] Project 4: Edit mode working

Then Part 2 can be implemented as an enhancement.
