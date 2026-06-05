"""Atmospheric rendering subsystem.

This module provides physically-based atmospheric scattering using the
Bruneton 2017 precomputed atmospheric scattering approach. It generates
lookup tables for:
- Transmittance (optical depth along view rays)
- Sky-View (single scattering for the sky dome)
- Aerial Perspective (in-scattering for distant objects)

See RENDERING_CONTEXT.md for full specification.
"""

from engine.rendering.atmosphere.bruneton_lut import (
    # Parameters
    AtmosphereParams,
    # LUT dimensions
    LUTDimensions,
    # Phase functions
    rayleigh_phase,
    cornette_shanks_phase,
    # Core computation
    compute_optical_depth,
    compute_transmittance,
    # Validation
    validate_transmittance_lut,
    validate_sky_view_lut,
    validate_aerial_perspective_lut,
    # Generator class
    BrunetonLUTGenerator,
)

__all__ = [
    # Parameters
    "AtmosphereParams",
    # LUT dimensions
    "LUTDimensions",
    # Phase functions
    "rayleigh_phase",
    "cornette_shanks_phase",
    # Core computation
    "compute_optical_depth",
    "compute_transmittance",
    # Validation
    "validate_transmittance_lut",
    "validate_sky_view_lut",
    "validate_aerial_perspective_lut",
    # Generator class
    "BrunetonLUTGenerator",
]
