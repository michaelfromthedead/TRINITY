# PHASE 2 ARCHITECTURE: PBR Model and Texture Pipeline

## Overview

Phase 2 focuses on the PBR (Physically Based Rendering) model implementation, including the metallic-roughness workflow, texture sets, and workflow variants.

## Components

### PBRParameters
- Core metallic-roughness dataclass
- Fields: base_color, metallic, roughness, normal, emissive, ao, opacity
- Validation for parameter ranges (metallic 0-1, roughness 0-1)

### PBRMaterial
- Component wrapper for PBRParameters
- Tracked descriptors for GPU binding
- Dirty flag integration
- Change callbacks for reactive updates

### PBRTextureSet
- Texture bindings for PBR workflow
- Channel configuration (which texture channels map to which parameters)
- Support for packed textures (ORM: occlusion-roughness-metallic)

### PBRWorkflow
- Enum: METALLIC_ROUGHNESS, SPECULAR_GLOSSINESS
- Determines parameter interpretation
- Affects shader generation

## Architecture Decisions

### AD-4: Metallic-Roughness as Primary Workflow
**Decision**: Default to metallic-roughness workflow, specular-glossiness as opt-in.
**Rationale**: Industry standard (glTF, Unreal, Unity default). Simpler energy conservation.
**Consequences**: Legacy assets using spec-gloss need conversion or explicit workflow selection.

### AD-5: Packed Texture Support
**Decision**: Support ORM (occlusion-roughness-metallic) packed textures.
**Rationale**: Reduces texture samples per pixel. Common in production assets.
**Consequences**: Channel swizzling logic in shader generation.

### AD-6: Dirty Flag Granularity for PBR
**Decision**: Separate dirty flags for parameters vs textures.
**Rationale**: Texture rebinding is more expensive than uniform update.
**Consequences**: Renderer can batch uniform updates separately from texture binds.

## Texture Binding Layout

| Slot | Texture | Channels |
|------|---------|----------|
| 0 | Base Color | RGB: albedo, A: opacity |
| 1 | Normal | RG: normal XY (reconstruct Z) |
| 2 | ORM | R: occlusion, G: roughness, B: metallic |
| 3 | Emissive | RGB: emissive color |

## Validation Strategy

1. Unit test PBRParameters validation (range clamping, normalization)
2. Test PBRTextureSet channel mapping
3. Verify dirty flag behavior on parameter and texture changes
4. Integration test with texture table binding
