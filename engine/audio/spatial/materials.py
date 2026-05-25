"""Acoustic Materials System.

Defines how different materials affect sound propagation:
- Frequency-dependent absorption
- Reflection coefficients
- Transmission coefficients
- Scattering properties
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from engine.audio.spatial.config import TRANSMISSION_LOSS


class MaterialType(Enum):
    """Standard acoustic material types."""

    CONCRETE = auto()
    """Concrete, cement, stone."""

    BRICK = auto()
    """Brick, masonry."""

    WOOD = auto()
    """Wood panels, flooring."""

    GLASS = auto()
    """Glass windows, mirrors."""

    METAL = auto()
    """Metal surfaces, steel."""

    CARPET = auto()
    """Carpet, rugs."""

    FABRIC = auto()
    """Fabric, curtains, upholstery."""

    TILE = auto()
    """Tile, marble, ceramic."""

    DRYWALL = auto()
    """Drywall, plasterboard."""

    GRASS = auto()
    """Grass, outdoor ground."""

    GRAVEL = auto()
    """Gravel, loose stones."""

    WATER = auto()
    """Water surface."""

    SNOW = auto()
    """Snow, ice."""

    ACOUSTIC_FOAM = auto()
    """Acoustic foam panels."""

    ACOUSTIC_TILE = auto()
    """Acoustic ceiling tiles."""

    CUSTOM = auto()
    """Custom user-defined material."""


# Standard frequency bands for acoustic analysis (Hz)
FREQUENCY_BANDS: Tuple[float, ...] = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)


@dataclass
class AcousticMaterial:
    """Acoustic properties of a material."""

    name: str
    """Human-readable material name."""

    material_type: MaterialType
    """Material type classification."""

    # Absorption coefficients (0-1) at standard frequency bands
    absorption_125hz: float = 0.1
    """Absorption at 125 Hz."""

    absorption_250hz: float = 0.1
    """Absorption at 250 Hz."""

    absorption_500hz: float = 0.1
    """Absorption at 500 Hz."""

    absorption_1000hz: float = 0.1
    """Absorption at 1000 Hz."""

    absorption_2000hz: float = 0.1
    """Absorption at 2000 Hz."""

    absorption_4000hz: float = 0.1
    """Absorption at 4000 Hz."""

    reflection: float = 0.9
    """Overall reflection coefficient (0-1)."""

    transmission: float = 0.0
    """Transmission coefficient - how much passes through (0-1)."""

    scattering: float = 0.1
    """Scattering coefficient - diffuse vs specular (0-1)."""

    density: float = 1000.0
    """Material density in kg/m^3 (affects transmission)."""

    def __post_init__(self) -> None:
        """Validate and clamp values."""
        self.absorption_125hz = max(0.0, min(1.0, self.absorption_125hz))
        self.absorption_250hz = max(0.0, min(1.0, self.absorption_250hz))
        self.absorption_500hz = max(0.0, min(1.0, self.absorption_500hz))
        self.absorption_1000hz = max(0.0, min(1.0, self.absorption_1000hz))
        self.absorption_2000hz = max(0.0, min(1.0, self.absorption_2000hz))
        self.absorption_4000hz = max(0.0, min(1.0, self.absorption_4000hz))
        self.reflection = max(0.0, min(1.0, self.reflection))
        self.transmission = max(0.0, min(1.0, self.transmission))
        self.scattering = max(0.0, min(1.0, self.scattering))
        self.density = max(1.0, self.density)

    def get_absorption(self, frequency: float) -> float:
        """Get interpolated absorption coefficient at a specific frequency.

        Args:
            frequency: Frequency in Hz.

        Returns:
            Absorption coefficient (0-1).
        """
        bands = [
            (125.0, self.absorption_125hz),
            (250.0, self.absorption_250hz),
            (500.0, self.absorption_500hz),
            (1000.0, self.absorption_1000hz),
            (2000.0, self.absorption_2000hz),
            (4000.0, self.absorption_4000hz),
        ]

        # Handle frequencies outside range
        if frequency <= bands[0][0]:
            return bands[0][1]
        if frequency >= bands[-1][0]:
            return bands[-1][1]

        # Linear interpolation between bands
        for i in range(len(bands) - 1):
            if bands[i][0] <= frequency <= bands[i + 1][0]:
                t = (frequency - bands[i][0]) / (bands[i + 1][0] - bands[i][0])
                return bands[i][1] + t * (bands[i + 1][1] - bands[i][1])

        return 0.1

    def get_absorption_coefficients(self) -> List[Tuple[float, float]]:
        """Get all absorption coefficients as (frequency, coefficient) pairs.

        Returns:
            List of (frequency, absorption) tuples.
        """
        return [
            (125.0, self.absorption_125hz),
            (250.0, self.absorption_250hz),
            (500.0, self.absorption_500hz),
            (1000.0, self.absorption_1000hz),
            (2000.0, self.absorption_2000hz),
            (4000.0, self.absorption_4000hz),
        ]

    @property
    def average_absorption(self) -> float:
        """Get average absorption across all frequency bands.

        Returns:
            Average absorption coefficient.
        """
        return (
            self.absorption_125hz +
            self.absorption_250hz +
            self.absorption_500hz +
            self.absorption_1000hz +
            self.absorption_2000hz +
            self.absorption_4000hz
        ) / 6.0

    @property
    def nrc(self) -> float:
        """Calculate Noise Reduction Coefficient (NRC).

        NRC is the average of absorption at 250, 500, 1000, and 2000 Hz,
        rounded to the nearest 0.05.

        Returns:
            NRC value.
        """
        avg = (
            self.absorption_250hz +
            self.absorption_500hz +
            self.absorption_1000hz +
            self.absorption_2000hz
        ) / 4.0
        return round(avg * 20) / 20  # Round to nearest 0.05

    def get_reflection_at_frequency(self, frequency: float) -> float:
        """Get reflection coefficient at a specific frequency.

        Accounts for frequency-dependent absorption.

        Args:
            frequency: Frequency in Hz.

        Returns:
            Reflection coefficient (0-1).
        """
        absorption = self.get_absorption(frequency)
        # Reflection = 1 - absorption (simplified model)
        return max(0.0, min(1.0, 1.0 - absorption)) * self.reflection


# Preset material definitions
MATERIAL_PRESETS: Dict[MaterialType, AcousticMaterial] = {
    MaterialType.CONCRETE: AcousticMaterial(
        name="Concrete",
        material_type=MaterialType.CONCRETE,
        absorption_125hz=0.01, absorption_250hz=0.01,
        absorption_500hz=0.02, absorption_1000hz=0.02,
        absorption_2000hz=0.02, absorption_4000hz=0.03,
        reflection=0.98, transmission=0.0, scattering=0.10,
        density=2400.0
    ),
    MaterialType.BRICK: AcousticMaterial(
        name="Brick",
        material_type=MaterialType.BRICK,
        absorption_125hz=0.02, absorption_250hz=0.02,
        absorption_500hz=0.03, absorption_1000hz=0.04,
        absorption_2000hz=0.05, absorption_4000hz=0.07,
        reflection=0.95, transmission=0.0, scattering=0.15,
        density=1800.0
    ),
    MaterialType.WOOD: AcousticMaterial(
        name="Wood",
        material_type=MaterialType.WOOD,
        absorption_125hz=0.15, absorption_250hz=0.11,
        absorption_500hz=0.10, absorption_1000hz=0.07,
        absorption_2000hz=0.06, absorption_4000hz=0.07,
        reflection=0.85, transmission=0.05, scattering=0.15,
        density=600.0
    ),
    MaterialType.GLASS: AcousticMaterial(
        name="Glass",
        material_type=MaterialType.GLASS,
        absorption_125hz=0.35, absorption_250hz=0.25,
        absorption_500hz=0.18, absorption_1000hz=0.12,
        absorption_2000hz=0.07, absorption_4000hz=0.04,
        reflection=0.90, transmission=0.02, scattering=0.05,
        density=2500.0
    ),
    MaterialType.METAL: AcousticMaterial(
        name="Metal",
        material_type=MaterialType.METAL,
        absorption_125hz=0.01, absorption_250hz=0.01,
        absorption_500hz=0.01, absorption_1000hz=0.02,
        absorption_2000hz=0.02, absorption_4000hz=0.03,
        reflection=0.95, transmission=0.0, scattering=0.05,
        density=7800.0
    ),
    MaterialType.CARPET: AcousticMaterial(
        name="Carpet",
        material_type=MaterialType.CARPET,
        absorption_125hz=0.08, absorption_250hz=0.24,
        absorption_500hz=0.57, absorption_1000hz=0.69,
        absorption_2000hz=0.71, absorption_4000hz=0.73,
        reflection=0.30, transmission=0.0, scattering=0.60,
        density=200.0
    ),
    MaterialType.FABRIC: AcousticMaterial(
        name="Fabric/Curtain",
        material_type=MaterialType.FABRIC,
        absorption_125hz=0.07, absorption_250hz=0.31,
        absorption_500hz=0.49, absorption_1000hz=0.75,
        absorption_2000hz=0.70, absorption_4000hz=0.60,
        reflection=0.25, transmission=0.15, scattering=0.70,
        density=100.0
    ),
    MaterialType.TILE: AcousticMaterial(
        name="Tile/Marble",
        material_type=MaterialType.TILE,
        absorption_125hz=0.01, absorption_250hz=0.01,
        absorption_500hz=0.01, absorption_1000hz=0.02,
        absorption_2000hz=0.02, absorption_4000hz=0.02,
        reflection=0.95, transmission=0.0, scattering=0.05,
        density=2700.0
    ),
    MaterialType.DRYWALL: AcousticMaterial(
        name="Drywall/Plasterboard",
        material_type=MaterialType.DRYWALL,
        absorption_125hz=0.29, absorption_250hz=0.10,
        absorption_500hz=0.05, absorption_1000hz=0.04,
        absorption_2000hz=0.07, absorption_4000hz=0.09,
        reflection=0.85, transmission=0.10, scattering=0.10,
        density=800.0
    ),
    MaterialType.GRASS: AcousticMaterial(
        name="Grass/Ground",
        material_type=MaterialType.GRASS,
        absorption_125hz=0.15, absorption_250hz=0.25,
        absorption_500hz=0.40, absorption_1000hz=0.55,
        absorption_2000hz=0.60, absorption_4000hz=0.60,
        reflection=0.40, transmission=0.0, scattering=0.80,
        density=1500.0
    ),
    MaterialType.GRAVEL: AcousticMaterial(
        name="Gravel",
        material_type=MaterialType.GRAVEL,
        absorption_125hz=0.25, absorption_250hz=0.60,
        absorption_500hz=0.65, absorption_1000hz=0.70,
        absorption_2000hz=0.75, absorption_4000hz=0.80,
        reflection=0.25, transmission=0.0, scattering=0.90,
        density=1700.0
    ),
    MaterialType.WATER: AcousticMaterial(
        name="Water",
        material_type=MaterialType.WATER,
        absorption_125hz=0.01, absorption_250hz=0.01,
        absorption_500hz=0.01, absorption_1000hz=0.02,
        absorption_2000hz=0.02, absorption_4000hz=0.03,
        reflection=0.95, transmission=0.0, scattering=0.10,
        density=1000.0
    ),
    MaterialType.SNOW: AcousticMaterial(
        name="Snow",
        material_type=MaterialType.SNOW,
        absorption_125hz=0.45, absorption_250hz=0.75,
        absorption_500hz=0.90, absorption_1000hz=0.95,
        absorption_2000hz=0.95, absorption_4000hz=0.95,
        reflection=0.10, transmission=0.0, scattering=0.90,
        density=300.0
    ),
    MaterialType.ACOUSTIC_FOAM: AcousticMaterial(
        name="Acoustic Foam",
        material_type=MaterialType.ACOUSTIC_FOAM,
        absorption_125hz=0.15, absorption_250hz=0.35,
        absorption_500hz=0.65, absorption_1000hz=0.85,
        absorption_2000hz=0.90, absorption_4000hz=0.90,
        reflection=0.10, transmission=0.05, scattering=0.80,
        density=50.0
    ),
    MaterialType.ACOUSTIC_TILE: AcousticMaterial(
        name="Acoustic Ceiling Tile",
        material_type=MaterialType.ACOUSTIC_TILE,
        absorption_125hz=0.30, absorption_250hz=0.35,
        absorption_500hz=0.50, absorption_1000hz=0.65,
        absorption_2000hz=0.70, absorption_4000hz=0.65,
        reflection=0.35, transmission=0.05, scattering=0.60,
        density=350.0
    ),
}


class MaterialDatabase:
    """Database of acoustic materials."""

    def __init__(self) -> None:
        """Initialize the material database with presets."""
        self._materials: Dict[str, AcousticMaterial] = {}
        self._load_presets()

    def _load_presets(self) -> None:
        """Load all preset materials."""
        for material in MATERIAL_PRESETS.values():
            self._materials[material.name.lower()] = material

    def get(self, name: str) -> Optional[AcousticMaterial]:
        """Get material by name (case-insensitive).

        Args:
            name: Material name.

        Returns:
            AcousticMaterial or None if not found.
        """
        return self._materials.get(name.lower())

    def get_by_type(self, material_type: MaterialType) -> Optional[AcousticMaterial]:
        """Get preset material by type.

        Args:
            material_type: Material type enum value.

        Returns:
            AcousticMaterial or None if not found.
        """
        return MATERIAL_PRESETS.get(material_type)

    def register(self, material: AcousticMaterial) -> None:
        """Register a custom material.

        Args:
            material: Material to register.
        """
        self._materials[material.name.lower()] = material

    def unregister(self, name: str) -> bool:
        """Unregister a material.

        Args:
            name: Material name to remove.

        Returns:
            True if material was removed, False if not found.
        """
        key = name.lower()
        if key in self._materials:
            del self._materials[key]
            return True
        return False

    def list_materials(self) -> List[str]:
        """List all available material names.

        Returns:
            List of material names.
        """
        return list(self._materials.keys())

    def get_all(self) -> List[AcousticMaterial]:
        """Get all registered materials.

        Returns:
            List of all materials.
        """
        return list(self._materials.values())

    def calculate_room_rt60(
        self,
        volume: float,
        surface_areas: Dict[str, float]
    ) -> float:
        """Estimate RT60 (reverberation time) for a room using Sabine equation.

        RT60 = 0.161 * V / A
        where V is volume and A is total absorption area.

        Args:
            volume: Room volume in cubic meters.
            surface_areas: Dict mapping material names to surface area in m^2.

        Returns:
            Estimated RT60 in seconds.
        """
        if volume <= 0:
            return 0.0

        total_absorption = 0.0

        for material_name, area in surface_areas.items():
            material = self.get(material_name)
            if material is not None:
                total_absorption += area * material.average_absorption

        if total_absorption < 0.01:
            return 10.0  # Very reverberant room

        # Sabine equation
        rt60 = 0.161 * volume / total_absorption
        return min(rt60, 20.0)  # Cap at 20 seconds

    def calculate_room_rt60_eyring(
        self,
        volume: float,
        total_surface_area: float,
        average_absorption: float
    ) -> float:
        """Estimate RT60 using Eyring equation (more accurate for high absorption).

        RT60 = 0.161 * V / (-S * ln(1 - alpha))

        Args:
            volume: Room volume in cubic meters.
            total_surface_area: Total surface area in m^2.
            average_absorption: Average absorption coefficient.

        Returns:
            Estimated RT60 in seconds.
        """
        import math

        if volume <= 0 or total_surface_area <= 0:
            return 0.0

        average_absorption = max(0.001, min(0.999, average_absorption))

        try:
            rt60 = 0.161 * volume / (-total_surface_area * math.log(1.0 - average_absorption))
            return min(max(0.0, rt60), 20.0)
        except (ValueError, ZeroDivisionError):
            return 10.0


def create_custom_material(
    name: str,
    absorption_coeffs: Tuple[float, float, float, float, float, float],
    reflection: float = 0.5,
    transmission: float = 0.0,
    scattering: float = 0.2
) -> AcousticMaterial:
    """Create a custom acoustic material.

    Args:
        name: Material name.
        absorption_coeffs: Absorption at (125, 250, 500, 1000, 2000, 4000) Hz.
        reflection: Reflection coefficient.
        transmission: Transmission coefficient.
        scattering: Scattering coefficient.

    Returns:
        New AcousticMaterial instance.
    """
    return AcousticMaterial(
        name=name,
        material_type=MaterialType.CUSTOM,
        absorption_125hz=absorption_coeffs[0],
        absorption_250hz=absorption_coeffs[1],
        absorption_500hz=absorption_coeffs[2],
        absorption_1000hz=absorption_coeffs[3],
        absorption_2000hz=absorption_coeffs[4],
        absorption_4000hz=absorption_coeffs[5],
        reflection=reflection,
        transmission=transmission,
        scattering=scattering
    )


def get_transmission_loss_db(material: AcousticMaterial, thickness: float) -> float:
    """Calculate transmission loss through a material.

    Uses simplified mass law approximation.

    Args:
        material: The acoustic material.
        thickness: Material thickness in meters.

    Returns:
        Transmission loss in decibels.
    """
    import math

    # Get base transmission loss from config if available
    type_name = material.material_type.name.lower()
    base_loss = TRANSMISSION_LOSS.get(type_name, 20.0)

    # Scale by thickness (simplified)
    # Mass law: TL increases ~6 dB per doubling of mass
    reference_thickness = 0.1  # 10 cm reference
    thickness_factor = math.log2(max(0.01, thickness) / reference_thickness)
    thickness_loss = 6.0 * thickness_factor

    return max(0.0, base_loss + thickness_loss)


def calculate_absorption_area(
    materials: List[Tuple[AcousticMaterial, float]]
) -> float:
    """Calculate total absorption area (sabins).

    Args:
        materials: List of (material, area) tuples.

    Returns:
        Total absorption area in sabins (m^2).
    """
    total = 0.0
    for material, area in materials:
        total += material.average_absorption * area
    return total
