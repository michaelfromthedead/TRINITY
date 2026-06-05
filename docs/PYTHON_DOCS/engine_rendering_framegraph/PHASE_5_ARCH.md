# PHASE 5 ARCHITECTURE: Rust Bridge Validation

## Overview

Validate and complete the Python-to-Rust serialization path. The `serialize()` method outputs JSON matching Rust IR types; this phase ensures round-trip fidelity and integration with the renderer backend crate.

## Architecture Decisions

### ADR-FG-019: Allocation Data Required Before Serialize

**Decision**: `serialize()` requires allocation to be complete (Phase 2 must be done).

**Rationale**:
- Rust side needs concrete heap_id/offset to bind resources
- Serializing pending allocations would require Rust-side allocation
- Clean separation: Python allocates, Rust executes

**Consequences**:
- `serialize()` before `begin_frame()` raises error
- IR includes allocation handles as concrete values
- Rust trusts allocation data is valid

## Serialization Schema

```json
{
  "version": "1.0",
  "passes": [
    {
      "name": "GBuffer",
      "pass_type": "graphics",
      "reads": [],
      "writes": [
        {"resource": "Albedo", "state": "render_target", "subresource": null}
      ],
      "execution_order": 0,
      "queue": "graphics"
    }
  ],
  "resources": [
    {
      "name": "Albedo",
      "type": "texture_2d",
      "format": "rgba8_unorm",
      "width": 1920,
      "height": 1080,
      "allocation": {
        "heap_id": 0,
        "offset": 0,
        "size": 8294400
      }
    }
  ],
  "sync_points": [
    {"signal_queue": "compute", "wait_queue": "graphics", "fence_value": 1}
  ]
}
```
