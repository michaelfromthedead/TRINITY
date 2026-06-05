# Phase 4: Advanced Shading — Architecture

## Status: PYTHON MODELS ONLY

All 6 advanced shading models exist as Python parameter classes with mathematical implementations. No WGSL shader code exists for any model. No GPU implementation.

## Current Architecture

### Python Models (`engine/rendering/materials/advanced_models.py`)

All models share a common structure:

```
ShadingModelType (enum):
  SUBSURFACE | CLEAR_COAT | ANISOTROPY | SHEEN | IRIDESCENCE | TRANSMISSION
```

### Individual Model Classes

#### SubsurfaceScattering
- `SubsurfaceProfile` dataclass: name, scatter_radius, scatter_color, falloff_color, transmittance_color, boundary_color_bleed, curvature_scale
- `get_diffusion_profile(num_samples)` → Burley normalized diffusion weights
- `to_shader_data()` → Dict for shader uniform upload

#### ClearCoat
- Fields: intensity, roughness, IOR, base_color_mix
- No BRDF implementation

#### Anisotropy
- Fields: strength, angle
- No anisotropic GGX NDF

#### Sheen
- Fields: color, roughness, intensity
- No microfiber lobe

#### Iridescence
- Fields: intensity, IOR, thickness_min, thickness_max
- No thin-film interference

#### Transmission
- Fields: factor, IOR, roughness, thickness
- No refraction, no Beer's law

## Missing for GPU Implementation

For each model, the following are needed in WGSL:

1. **SSS** — Dual-pass screen-space: irradiance buffer, Burley separable blur (12-24 taps), importance-sampled kernel
2. **Clear coat** — Dual-layer BRDF: top-layer Schlick (F0=0.04, IOR=1.5), Fresnel-weighted blend
3. **Anisotropy** — Anisotropic GGX NDF with alpha_x/alpha_y, stretched tangent-space BRDF
4. **Sheen** — Microfiber retro-reflection lobe, third BRDF lobe, sheen_color tint
5. **Transmission** — Screen-space refraction, thickness map UV offset, Beer's law absorption
6. **Iridescence** — Thin-film interference: film phase, air-film Fresnel, interference color

All models require integration with quality tier const bool gating.

## Cross-References

- `engine/rendering/materials/advanced_models.py` — All 6 Python models with full parameter definitions (334 lines)
- `engine/rendering/materials/constants.py` — SSS, clear coat, anisotropy, sheen, iridescence, transmission parameter ranges
- `crates/renderer-backend/shaders/pbr.frag.wgsl` — Base WGSL shader where models would be added
- `GAP_4_SUMMARY.md` → Phase 4 task verification
