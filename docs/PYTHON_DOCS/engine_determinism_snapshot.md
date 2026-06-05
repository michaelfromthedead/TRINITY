# Investigation: engine/determinism/snapshot

## Summary
The `engine/determinism/snapshot/` directory is an **empty scaffold** containing only a 0-byte `__init__.py`. However, snapshot functionality IS implemented elsewhere: the `@snapshot` decorator in `trinity/decorators/data_flow.py` provides working ring-buffer state capture with `snapshot_save()` and `snapshot_restore()` methods. The design intent (accordion strategy, ContentStore integration, hierarchical checksums) is fully documented in the 92KB `DETERMINISM_CONTEXT.md` but not yet implemented in this location.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 0 | EMPTY | Placeholder only |

## Related Implementation (Outside Directory)
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `trinity/decorators/data_flow.py` | 456 | REAL | Contains working `@snapshot` decorator |
| `engine/determinism/DETERMINISM_CONTEXT.md` | ~2000 | DESIGN DOC | Full architecture spec (92KB) |

## Snapshot Components

### Implemented (in trinity/decorators/data_flow.py):
- `SnapshotConfig` dataclass (history_frames configuration)
- `@snapshot` decorator with validation
- `snapshot_save()` method - ring buffer state capture
- `snapshot_restore()` method - state restoration by frame index
- Integration with `@serializable` decorator (required dependency)

### Designed but NOT Implemented (from DETERMINISM_CONTEXT.md):
- Accordion snapshot strategy (dense recent, sparse old)
- ContentStore integration for efficient storage
- Hierarchical checksums (XOR-based per-entity/archetype/world)
- Delta compression for old snapshots
- Network rollback (GGPO-style) integration
- Frame-perfect replay seeking
- Divergence detection

## Verdict
**PARTIAL** - The `engine/determinism/snapshot/` directory itself is empty, but basic snapshot functionality exists in the decorator layer. The directory is a scaffold awaiting the full determinism system implementation.

## Evidence

### Empty Directory Contents:
```
engine/determinism/snapshot/
  __init__.py  (0 bytes)
```

### Working Decorator Implementation (data_flow.py lines 296-346):
```python
def _after_snapshot(target: Any, params: dict[str, Any]) -> Any:
    """Attach snapshot history and methods."""
    validate_target_type(target, "snapshot", ("class",))

    # Validate that @serializable is present
    if not hasattr(target, "_serializable"):
        raise TypeError(
            f"@snapshot requires @serializable to be applied first on {target.__name__}"
        )

    config = target._tags.get("snapshot_config")
    target._snapshot = True
    target._snapshot_history_frames = config.history_frames

    # Ring buffer implementation
    def snapshot_save(self) -> int:
        if not hasattr(self, "_snapshot_history"):
            self._snapshot_history = []
            self._snapshot_frame = 0

        state = self.__class__.serialize(self)
        if len(self._snapshot_history) >= self._snapshot_history_frames:
            self._snapshot_history.pop(0)

        self._snapshot_history.append(state)
        self._snapshot_frame += 1
        return self._snapshot_frame - 1

    def snapshot_restore(self, frame: int) -> bool:
        if not hasattr(self, "_snapshot_history"):
            return False
        if frame < 0 or frame >= len(self._snapshot_history):
            return False
        state = self._snapshot_history[frame]
        for key, value in state.items():
            if key not in ("__version__", "__type__"):
                setattr(self, key, value)
        return True

    target.snapshot_save = snapshot_save
    target.snapshot_restore = snapshot_restore
```

### Sibling Directories Also Empty:
- `engine/determinism/core/__init__.py` (0 bytes)
- `engine/determinism/network/__init__.py` (0 bytes)
- `engine/determinism/replay/__init__.py` (0 bytes)

## Conclusion
The snapshot directory is part of a scaffolded determinism layer. Basic snapshot functionality works via decorators, but the advanced features (accordion strategy, ContentStore, checksums, GGPO rollback) documented in DETERMINISM_CONTEXT.md remain unimplemented. This is a "design-first" area awaiting implementation.
