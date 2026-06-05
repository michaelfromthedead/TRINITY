"""Water rendering passes for TRINITY engine.

This module provides water rendering components including:
- WaterPass: Base water rendering pass
- OceanRenderer: FFT-based ocean wave simulation
- WaterMaterial: Physically-based water material
- DeterministicGerstnerWave: Fixed32 Gerstner wave simulation (T-CC-2.2)

Stub created by T-ENV-1.12, expanded by:
- T-ENV-1.7: Ocean wave simulation
- T-ENV-1.8: Water material system
- T-CC-2.2: Deterministic Gerstner waves with Fixed32
"""

from .water_pass import WaterPass
from .ocean import OceanRenderer
from .water_material import WaterMaterial
from .deterministic_gerstner import (
    DeterministicGerstnerWave,
    Fixed32WaveParams,
    Fixed32Vec2,
    Fixed32Vec3,
    GerstnerWaveResult,
    compute_gerstner_wave,
    fixed32_sin,
    fixed32_cos,
    fixed32_sincos,
    FIXED32_ZERO,
    FIXED32_ONE,
    FIXED32_PI,
    FIXED32_TWO_PI,
)

__all__ = [
    # Base water rendering
    "WaterPass",
    "OceanRenderer",
    "WaterMaterial",
    # Deterministic Gerstner (T-CC-2.2)
    "DeterministicGerstnerWave",
    "Fixed32WaveParams",
    "Fixed32Vec2",
    "Fixed32Vec3",
    "GerstnerWaveResult",
    "compute_gerstner_wave",
    "fixed32_sin",
    "fixed32_cos",
    "fixed32_sincos",
    "FIXED32_ZERO",
    "FIXED32_ONE",
    "FIXED32_PI",
    "FIXED32_TWO_PI",
]
