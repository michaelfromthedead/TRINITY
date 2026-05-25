# INVENTORY: engine_animation_crowds_facial

**RDC Workflow Output**
**Generated:** 2026-05-23
**Subsystem:** Animation - Crowds + Facial

---

## Source Documents (Temporal Order)

| # | Document | Lines | Investigation Date | Purpose |
|---|----------|-------|-------------------|---------|
| 1 | `engine_animation_crowds.md` | 184 | 2026-05-22 | Detailed investigation of crowds subsystem |
| 2 | `engine_animation_facial.md` | 103 | 2026-05-22 | Detailed investigation of facial subsystem |
| 3 | `engine_animation_crowds_facial.md` | 262 | 2026-05-22 | Combined cross-module integration report |

---

## Temporal Ordering Rationale

All three documents were produced on 2026-05-22 as part of the 4-worker archaeological swarm investigation. Documents 1 and 2 represent individual subsystem deep-dives, while document 3 represents the synthesis and cross-module integration analysis. The temporal order reflects increasing synthesis level.

---

## Document Classification

| Document | Status | Evidence Quality |
|----------|--------|-----------------|
| `engine_animation_crowds.md` | PRIMARY | Direct code analysis with line references |
| `engine_animation_facial.md` | PRIMARY | Direct code analysis with line references |
| `engine_animation_crowds_facial.md` | SYNTHESIS | Cross-module integration, combined analysis |

---

## Files Analyzed (from source documents)

### Crowds Subsystem (4 files, 2,237 lines)
- `__init__.py` (63 lines)
- `animation_texture.py` (511 lines)
- `crowd_behavior.py` (711 lines)
- `crowd_lod.py` (497 lines)
- `crowd_renderer.py` (459 lines)

### Facial Subsystem (8 files, 5,233 lines)
- `__init__.py` (141 lines)
- `blend_shapes.py` (724 lines)
- `facs.py` (749 lines)
- `lip_sync.py` (903 lines)
- `eye_animation.py` (721 lines)
- `face_rig.py` (756 lines)
- `face_capture.py` (978 lines)
- `config.py` (261 lines)

**Combined Total: 12 files, ~7,470 lines of REAL implementation**
