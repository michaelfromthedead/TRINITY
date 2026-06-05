# PHASE 5 ARCHITECTURE: Advanced Shading Models

## Overview

Phase 5 covers the advanced shading models that extend beyond standard PBR: subsurface scattering, clearcoat, anisotropy, sheen, iridescence, and transmission.

## Components

### SubsurfaceScattering
- Burley diffusion profiles
- Presets: Skin, Wax, Jade, Milk
- Parameters: scattering_color, scattering_radius, falloff
- Requires blur pass for realistic results

### ClearCoat
- Secondary specular layer on top of base material
- IOR-based F0 calculation
- Parameters: clearcoat_intensity, clearcoat_roughness, clearcoat_normal

### Anisotropy
- Directional roughness for brushed metal, hair
- GGX parameterization
- Parameters: anisotropy_strength, anisotropy_rotation, tangent_map

### Sheen
- Fabric and velvet effects
- Edge-scattered light simulation
- Parameters: sheen_color, sheen_roughness

### Iridescence
- Thin film interference
- Thickness-dependent color shift
- Parameters: iridescence_intensity, iridescence_ior, iridescence_thickness

### Transmission
- Glass, liquid, and translucent materials
- Beer-Lambert attenuation
- Parameters: transmission, thickness, attenuation_color, attenuation_distance

## Architecture Decisions

### AD-14: Dataclass Parameters
**Decision**: Each advanced model uses a dedicated dataclass for parameters.
**Rationale**: Strong typing, validation, serialization support.
**Consequences**: More classes, but clear API boundaries.

### AD-15: Preset System
**Decision**: Common configurations are available as presets.
**Rationale**: Artists can start from known-good values.
**Consequences**: Presets must be curated and documented.

### AD-16: Modular Combination
**Decision**: Advanced models can be combined with base PBR.
**Rationale**: Real materials often combine effects (car paint = metallic + clearcoat).
**Consequences**: Shader complexity scales with enabled features.

### AD-17: Performance Tiers
**Decision**: Advanced models can be disabled per-platform.
**Rationale**: SSS blur, transmission refraction are expensive.
**Consequences**: Materials must have graceful fallbacks.

## Shading Model Complexity

| Model | Extra Passes | GPU Cost | Use Cases |
|-------|--------------|----------|-----------|
| SSS | Blur pass | High | Skin, wax, candles |
| ClearCoat | None | Low | Car paint, lacquer |
| Anisotropy | None | Low | Brushed metal, hair |
| Sheen | None | Low | Fabric, velvet |
| Iridescence | None | Medium | Soap bubbles, oil |
| Transmission | Refraction | High | Glass, liquids |

## Integration Points

- `engine/rendering/materials/pbr_model.py` - Base PBR extension
- `engine/rendering/frame_graph` - Additional render passes
- Lighting shaders - Extended BRDF evaluation

## Validation Strategy

1. Unit test parameter validation for each model
2. Test preset application
3. Visual regression test against reference renders
4. Performance profiling per model
