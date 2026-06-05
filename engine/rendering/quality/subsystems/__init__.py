"""Quality tier configurations for rendering subsystems (T-CC-0.5, T-CC-0.6)."""

from .atmosphere import AtmosphereCapabilities
from .demoscene import DemosceneCapabilities
from .gi import GICapabilities
from .lighting import LightingCapabilities
from .materials import MaterialsCapabilities
from .particles import ParticlesCapabilities
from .postprocess import PostProcessCapabilities
from .raytracing import RayTracingCapabilities
from .reflections import ReflectionsCapabilities
from .shadows import ShadowsCapabilities
from .terrain import TerrainCapabilities

__all__ = [
    "AtmosphereCapabilities",
    "DemosceneCapabilities",
    "GICapabilities",
    "LightingCapabilities",
    "MaterialsCapabilities",
    "ParticlesCapabilities",
    "PostProcessCapabilities",
    "RayTracingCapabilities",
    "ReflectionsCapabilities",
    "ShadowsCapabilities",
    "TerrainCapabilities",
]
