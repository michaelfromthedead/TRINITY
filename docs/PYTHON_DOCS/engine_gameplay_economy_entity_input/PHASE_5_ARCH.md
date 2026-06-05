# PHASE 5 ARCHITECTURE: Serialization Completion

## Overview

Complete from_dict() methods for all serializable types to enable full save/load support.

## Serializable Types

### Economy Module

| Type | to_dict() Status | from_dict() Status | Priority |
|------|-----------------|-------------------|----------|
| ItemDefinition | Exists | **NEEDS WORK** | High |
| ItemInstance | Exists | **NEEDS WORK** | High |
| InventorySlot | Exists | **NEEDS WORK** | High |
| InventoryContainer | Exists | **NEEDS WORK** | High |
| Recipe | Exists | **NEEDS WORK** | Medium |
| CraftingQueue | Exists | **NEEDS WORK** | Medium |
| LootTable | Exists | **NEEDS WORK** | Low |
| EquipmentContainer | Exists | **NEEDS WORK** | High |
| StatModifier | Exists | **NEEDS WORK** | High |

### Entity Module

| Type | to_dict() Status | from_dict() Status | Priority |
|------|-----------------|-------------------|----------|
| Transform | Exists | **NEEDS WORK** | High |
| Actor (base) | Exists | **NEEDS WORK** | High |
| Pawn | Exists | **NEEDS WORK** | High |
| Character | Exists | **NEEDS WORK** | High |
| PrefabDefinition | Exists | **NEEDS WORK** | Medium |
| ControllerState | Exists | **NEEDS WORK** | Medium |

### Input Module

| Type | to_dict() Status | from_dict() Status | Priority |
|------|-----------------|-------------------|----------|
| ActionBinding | Exists | **NEEDS WORK** | Medium |
| AxisBinding | Exists | **NEEDS WORK** | Medium |
| InputSettings | Exists | **NEEDS WORK** | Medium |

## Architecture Decisions

### ADR-SER-1: Symmetric Serialization

Every from_dict() must be the exact inverse of to_dict():
```python
obj == Type.from_dict(obj.to_dict())
```

This is the primary acceptance criterion for all serialization work.

### ADR-SER-2: Version Field

All serialized dicts must include a `version` field for future migration:
```python
def to_dict(self) -> dict:
    return {
        "version": 1,
        "id": self.id,
        # ...
    }
```

### ADR-SER-3: Reference Resolution

References between objects (e.g., owner_id, parent_id) serialize as IDs, not embedded objects. Deserialization requires a resolution context:
```python
@classmethod
def from_dict(cls, data: dict, context: DeserializationContext) -> Self:
    owner = context.resolve_actor(data["owner_id"])
    # ...
```

### ADR-SER-4: Enum Serialization

Enums serialize as their name (string), not value:
```python
# Correct
{"rarity": "EPIC"}

# Incorrect
{"rarity": 3}
```

This enables enum reordering without breaking saves.

### ADR-SER-5: Optional Field Handling

Optional fields serialize as null (not omitted) for explicit presence tracking:
```python
# Correct
{"parent_id": null}

# Incorrect (field omitted)
{}
```

## Implementation Pattern

```python
@classmethod
def from_dict(cls, data: dict, context: DeserializationContext = None) -> Self:
    version = data.get("version", 1)
    if version > CURRENT_VERSION:
        raise VersionError(f"Cannot load version {version}, max supported is {CURRENT_VERSION}")
    
    # Migration for old versions
    if version < CURRENT_VERSION:
        data = cls._migrate(data, version)
    
    # Construct object
    return cls(
        id=data["id"],
        name=data["name"],
        # ... other fields
    )

@classmethod
def _migrate(cls, data: dict, from_version: int) -> dict:
    """Migrate data from old version to current."""
    if from_version == 1:
        # v1 -> v2 migration
        data["new_field"] = data.get("old_field", DEFAULT_VALUE)
    return data
```

## Test Requirements

Each from_dict() implementation must have:
1. Round-trip test: `obj == Type.from_dict(obj.to_dict())`
2. Minimal data test: only required fields present
3. All optional fields test: every optional field present
4. Version migration test: old version data loads correctly
5. Invalid data test: garbage data raises appropriate error

## Risks

| Risk | Mitigation |
|------|------------|
| Breaking existing saves | Version field + migration code |
| Reference cycles | ID-based references, not embedded |
| Missing fields in old saves | Default values in from_dict |
| Type confusion | Explicit enum/type serialization |
