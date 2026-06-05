# TRINITY Engine

Beyond-SOTA game engine with Python algorithms + Rust GPU backend.

---

## Quick Status

| Layer | Status | Lines |
|-------|--------|-------|
| Python Algorithms | ✅ Complete | 600,000+ |
| Rust Backend | ⚠️ 18% wired | 140,000 |
| Python↔Rust Bridge | ✅ Complete | GAPSET_3 |
| Rendering Pipeline | ❌ Blocked | Needs GAPSET_1 |

**Current work:** GAPSET_1_CORE (49%) — ThreadPool, JobGraph, Scheduler

See [`docs/STATUS.md`](docs/STATUS.md) for detailed progress.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PYTHON ENGINE (600K lines)                │
│  engine/rendering/   Visibility buffer, DDGI, PBR, VFX     │
│  engine/simulation/  GJK/EPA, XPBD, fluids, cloth          │
│  engine/animation/   FABRIK/CCD IK, motion matching        │
│  engine/gameplay/    GAS abilities, behavior trees          │
│  engine/audio/       Spatial audio, DSP, mixing             │
├─────────────────────────────────────────────────────────────┤
│                    BRIDGE (GAPSET_3 ✅)                      │
│  omega crate         PyO3 bindings, type/data/command       │
├─────────────────────────────────────────────────────────────┤
│                    RUST BACKEND (140K lines)                │
│  renderer-backend/   Frame graph, executor, RHI             │
│  gpu_driven/         Material, mesh, texture tables         │
│  wgpu                Vulkan/Metal/DX12/WebGPU               │
└─────────────────────────────────────────────────────────────┘
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/STATUS.md`](docs/STATUS.md) | Current implementation state |
| [`docs/architecture/ARCHITECTURE_INVESTIGATION_REPORT.md`](docs/architecture/ARCHITECTURE_INVESTIGATION_REPORT.md) | Complete architectural analysis |
| [`docs/DESIGN_DOCS_INDEX.md`](docs/DESIGN_DOCS_INDEX.md) | Index of 464 design documents |
| [`docs/investigations/WGPU_EXECUTION_PLAN.md`](docs/investigations/WGPU_EXECUTION_PLAN.md) | wgpu wiring roadmap |

### Specifications (14 CONTEXT.md files)

| Spec | Location | Lines |
|------|----------|-------|
| Rendering | `engine/rendering/RENDERING_CONTEXT.md` | 1,360 |
| Platform/RHI | `engine/platform/PLATFORM_CONTEXT.md` | 1,256 |
| Simulation | `engine/simulation/SIMULATION_CONTEXT.md` | 1,473 |
| Gameplay | `engine/gameplay/GAMEPLAY_CONTEXT.md` | 1,191 |
| Tooling | `engine/tooling/TOOLING_CONTEXT.md` | 1,968 |
| + 9 more | `engine/*/` | ~10,000 |

---

## Build

```bash
# Python (requires 3.13)
uv run pytest tests/

# Rust backend
cargo build --release
cargo test

# Bridge
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p omega --features pyo3
```

---

## Project Structure

```
TRINITY/
├── engine/           # Python frontend (982 files, 600K lines)
│   ├── rendering/    # Visibility buffer, DDGI, PBR, post-process
│   ├── simulation/   # Physics: rigid, cloth, fluid, vehicles
│   ├── animation/    # IK, motion matching, facial
│   ├── gameplay/     # GAS abilities, behavior trees
│   ├── audio/        # Spatial audio, DSP
│   └── ...           # 11 more subsystems
├── crates/           # Rust backend (130 files, 140K lines)
│   ├── omega/        # Math library + PyO3 bridge
│   └── renderer-backend/
│       ├── frame_graph/  # IR, compiler, barriers
│       ├── gpu_driven/   # Material/mesh tables
│       └── shaders/      # WGSL shaders
├── tests/            # Test suite (929 files)
└── docs/             # 884 markdown files
    ├── gap_sets/     # 20 Rust implementation gapsets
    └── phase_output/ # Python subsystem docs
```

---

## Unreal Engine 5 Parallels

| UE5 Feature | TRINITY Equivalent | Status |
|-------------|-------------------|--------|
| Nanite | Visibility Buffer | Python ✅, Rust wiring needed |
| Lumen | DDGI + gi_lumen.py | Python ✅, Lumen BUILD TARGET |
| Virtual Shadow Maps | virtual_shadow_maps.py | BUILD TARGET |
| Niagara | VFX Graph | Python ✅ (5,982 lines) |
| Blueprint | FlowForge | Spec only |
| GAS | Ability System | Python ✅ (3,136 lines) |

---

## Contributing

See [`CLAUDE.md`](CLAUDE.md) for development guidelines.

---

*Last updated: 2026-05-24*
