# PHASE 4 ARCHITECTURE: Material Functions Library

## Overview

Phase 4 covers the 20+ reusable material functions that can be embedded in generated shaders. These are pre-written GLSL snippets that implement common shading operations.

## Function Categories

### Fresnel Functions
- `Fresnel(viewDir, normal, power)` - Basic Fresnel effect
- `FresnelSchlick(cosTheta, F0)` - Schlick approximation
- `FresnelSchlickRoughness(cosTheta, F0, roughness)` - Roughness-aware Fresnel

### Normal Blending
- `NormalBlend(base, detail)` - Whiteout blending
- `NormalBlendRNM(base, detail)` - Reoriented normal mapping

### Parallax Mapping
- `ParallaxOffset(heightmap, uv, viewDir, scale)` - Simple offset
- `ParallaxOcclusionMapping(heightmap, uv, viewDir, layers, scale)` - POM with ray marching

### Triplanar Projection
- `TriplanarSample(texture, worldPos, normal, sharpness)` - Seamless texture projection

### Color Operations
- `sRGBToLinear(color)` - Gamma decode
- `LinearToSRGB(color)` - Gamma encode
- `Luminance(color)` - Perceptual brightness
- `Saturation(color, amount)` - Color saturation adjustment
- `Contrast(color, amount)` - Contrast adjustment

### Procedural Noise
- `ValueNoise(uv)` - Simple value noise
- `Voronoi(uv, randomness)` - Voronoi cells
- `GradientNoise(uv)` - Perlin-style gradient noise

### Procedural Patterns
- `Checkerboard(uv, scale)` - Checkerboard pattern
- `RadialGradient(uv, center, radius)` - Radial gradient

### Masking Functions
- `BoxMask(pos, min, max, falloff)` - Box-shaped mask
- `SphereMask(pos, center, radius, falloff)` - Sphere-shaped mask

### Blend Modes
- `BlendOverlay(base, blend)` - Overlay blend
- `BlendSoftLight(base, blend)` - Soft light blend

## Architecture Decisions

### AD-11: Embedded GLSL Strings
**Decision**: Functions are stored as GLSL string literals in Python.
**Rationale**: Simple packaging, no external file dependencies.
**Consequences**: Functions are not hot-reloadable individually.

### AD-12: Dependency Declaration
**Decision**: Functions declare dependencies on other functions.
**Rationale**: GraphCompiler can include only needed functions.
**Consequences**: Functions must be aware of their dependencies.

### AD-13: Uniform Injection
**Decision**: Functions can declare required uniforms.
**Rationale**: Some functions need external parameters.
**Consequences**: Uniform names must be coordinated to avoid conflicts.

## Function Structure

```python
def create_fresnel_function() -> MaterialFunction:
    code = """
// Fresnel effect using Schlick approximation
float Fresnel(vec3 viewDir, vec3 normal, float power) {
    float NdotV = max(dot(normal, viewDir), 0.0);
    return pow(1.0 - NdotV, power);
}
"""
    return MaterialFunction(
        name="Fresnel",
        code=code,
        inputs=["viewDir:vec3", "normal:vec3", "power:float"],
        outputs=["result:float"],
        dependencies=[]
    )
```

## Integration Points

- `engine/rendering/materials/material_graph.py` - FunctionCallNode
- `engine/rendering/materials/shader_compiler.py` - Include resolution
- Generated shaders - Direct GLSL inclusion

## Validation Strategy

1. Unit test each function compiles in isolation
2. Test function dependency resolution
3. Validate GLSL syntax with glslang
4. Visual regression test function outputs
