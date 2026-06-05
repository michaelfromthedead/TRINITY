# Phase 1: Compiler Foundation + IR -- Architecture

## Overview
Core data structures for the frame graph compiler. Defines the intermediate representation (IR) types that all subsequent compiler phases consume and produce. Covers S1-G1 (Frame Graph Compiler) and S1-G2 (View Trait).

## Key Data Structures

### Rust (`mod.rs` lines 688-1107)

| Struct/Enum | File | Description |
|-------------|------|-------------|
| `ResourceHandle(u32)` | mod.rs:42 | Opaque handle; `NONE = u32::MAX` |
| `PassIndex(usize)` | mod.rs:65 | Pass position in compilation array |
| `PassType { Graphics, Compute, Copy, RayTracing }` | mod.rs:83 | Workload classification |
| `ResourceAccess { Read, Write, ReadWrite }` | mod.rs:122 | Per-resource access pattern |
| `ResourceAccessSet { reads, writes }` | mod.rs:169 | Complete access set of a pass |
| `ColorAttachment` | mod.rs:229 | Render target binding (resource, mip, layer, load/store ops, clear color) |
| `DepthStencilAttachment` | mod.rs:278 | Depth/stencil binding (dual load/store ops, clear values, test/write flags) |
| `AttachmentLoadOp { Load, Clear, DontCare }` | mod.rs:340 | Attachment load behavior |
| `AttachmentStoreOp { Store, DontCare }` | mod.rs:361 | Attachment store behavior |
| `InstanceSource { Direct, Indirect, Mesh }` | mod.rs:385 | Geometry instance specification |
| `DispatchSource { Direct, Indirect }` | mod.rs:472 | Compute dispatch specification |
| `ViewType (9 variants)` | mod.rs:523 | Shader-visible binding type |
| `TextureDesc` | mod.rs:566 | 2D texture dimensions + format |
| `Texture3DDesc` | mod.rs:595 | 3D texture dimensions + format |
| `BufferDesc` | mod.rs:619 | Buffer size + usage |
| `ResourceDesc { Texture2D, Texture3D, TextureCube, Buffer }` | mod.rs:641 | Resource kind discriminator |
| `ResourceLifetime { Transient, Imported }` | mod.rs:666 | Lifetime management flag |
| `IrResource` | mod.rs:695 | Full resource IR (handle, name, desc, lifetime, initial_state) |
| `IrPass` | mod.rs:765 | Full pass IR (index, name, type, access_set, attachments, instance/dispatch source, view_type, tags) |
| `IrEdge` | mod.rs:1071 | Dependency edge (from, to, resource, edge_type) |

### Python (`python.rs`)

| Struct | Description |
|--------|-------------|
| `PyPassNode` | Python-side pass declaration (name, pass_type, color_attachments, depth_stencil, reads, writes, instance_source, dispatch_source, view_type) |
| `PyColorAttachment` | (resource handle, load_op string, store_op string) |
| `PyDepthStencilAttachment` | (resource, depth_load_op, depth_store_op, stencil_load_op, stencil_store_op) |
| `PyDispatchSource` | (kind string) |
| `PyInstanceSource` | (kind string) |
| `PyViewType` | (kind string) |
| `PyPassType` | string -> enum mapping |
| `ConversionError` | 13 error variants with Display |

## Conversion Flow

```
Python FrameGraph -> PyPassNode -> TryFrom<PyPassNode> -> IrPass
                                        |
                                   Validation:
                                   1. Name non-empty
                                   2. Graphics -> >=1 color attachment
                                   3. Compute -> no attachments, dispatch required
                                   4. RayTracing/Copy -> no attachments
                                   5. All handles != NONE
                                   6. Load/store ops valid strings
                                   7. View type valid string
                                   8. Instance/dispatch source valid string
```

## Implementation Status

| Task | Status | Location |
|------|--------|----------|
| T-FG-1.1 | [x] | mod.rs:688-1107 |
| T-FG-1.2 | [x] | python.rs:327-482 |
| T-FG-1.3 | [-] | Not implemented |
| T-FG-1.4 | [-] | Not implemented |
| T-FG-1.5 | [-] | Not implemented |
| T-FG-1.6 | [~] | frame_graph.py:566-597 (Python-only compile) |
| T-FG-1.7 | [-] | Not implemented |
| T-FG-1.8 | [~] | blackbox_frame_graph_ir.rs (4 integration tests) |

## Gaps & Risks

1. **Missing PyResourceDesc->IrResource conversion** -- Resource conversion exists only via JSON deserialization, not through a typed conversion struct
2. **View trait completely absent** -- No `bind()` method, no trait objects, no `CameraView`. `ViewType` enum is sufficient for current needs but blocks custom shader binding layouts
3. **No PyO3 compile function** -- Python passes cannot call the Rust compiler directly
4. **ConversionError lacks format compatibility validation** -- No format field range or compatibility checking
