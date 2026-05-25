"""
Aerodynamic forces for cloth simulation.

Implements wind forces including:
- Per-triangle drag and lift
- Turbulence via noise functions
- Wind influence maps
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from .cloth_simulation import ClothMesh, ClothParticle, ClothTriangle


@dataclass
class WindSettings:
    """Configuration for wind forces."""

    direction: NDArray[np.float32] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    strength: float = 1.0
    drag_coefficient: float = 0.5
    lift_coefficient: float = 0.2
    turbulence_strength: float = 0.3
    turbulence_frequency: float = 2.0
    turbulence_octaves: int = 3


class WindForce:
    """
    Wind force generator for cloth simulation.

    Applies aerodynamic drag and lift forces to cloth triangles
    based on wind velocity and triangle orientation.
    """

    def __init__(self, settings: Optional[WindSettings] = None) -> None:
        """
        Initialize the wind force.

        Args:
            settings: Wind configuration settings
        """
        self.settings = settings or WindSettings()
        self._time: float = 0.0

        # Per-vertex wind influence (1.0 = full influence)
        self._vertex_influence: Optional[NDArray[np.float32]] = None

        # Turbulence seed for reproducibility
        self._turbulence_seed: int = 42

    def set_direction(self, direction: NDArray[np.float32]) -> None:
        """Set the wind direction (will be normalized)."""
        length = np.linalg.norm(direction)
        if length > 1e-8:
            self.settings.direction = (direction / length).astype(np.float32)

    def set_strength(self, strength: float) -> None:
        """Set the wind strength."""
        self.settings.strength = max(0.0, strength)

    def set_vertex_influence(
        self,
        influence: NDArray[np.float32],
    ) -> None:
        """
        Set per-vertex wind influence.

        Args:
            influence: Array of influence values (0-1) per vertex
        """
        self._vertex_influence = influence.astype(np.float32)

    def update(self, dt: float) -> None:
        """Update time for turbulence animation."""
        self._time += dt

    def compute_wind_force(
        self,
        mesh: ClothMesh,
        dt: float,
    ) -> None:
        """
        Compute and apply wind forces to cloth particles.

        Uses per-triangle aerodynamic model with drag and lift.

        Args:
            mesh: The cloth mesh
            dt: Timestep for integration
        """
        self.update(dt)

        particles = mesh.particles
        triangles = mesh.triangles

        # Pre-compute wind velocity with turbulence
        base_wind = self.settings.direction * self.settings.strength

        for tri in triangles:
            # Get triangle vertices
            p0 = particles[tri.p0]
            p1 = particles[tri.p1]
            p2 = particles[tri.p2]

            # Skip if all vertices are pinned
            if p0.inv_mass == 0 and p1.inv_mass == 0 and p2.inv_mass == 0:
                continue

            # Triangle center for turbulence sampling
            center = (p0.position + p1.position + p2.position) / 3.0

            # Get wind velocity at triangle
            wind_velocity = self._get_wind_at_position(center, base_wind)

            # Compute triangle normal and area
            edge1 = p1.position - p0.position
            edge2 = p2.position - p0.position
            cross = np.cross(edge1, edge2)
            area_x2 = float(np.linalg.norm(cross))

            if area_x2 < 1e-8:
                continue

            normal = (cross / area_x2).astype(np.float32)
            area = area_x2 * 0.5

            # Triangle velocity (average of vertex velocities)
            tri_velocity = (
                (p0.position - p0.prev_position)
                + (p1.position - p1.prev_position)
                + (p2.position - p2.prev_position)
            ) / (3.0 * dt) if dt > 0 else np.zeros(3, dtype=np.float32)

            # Relative wind velocity
            relative_wind = wind_velocity - tri_velocity
            relative_speed = float(np.linalg.norm(relative_wind))

            if relative_speed < 1e-8:
                continue

            # Wind direction
            wind_dir = relative_wind / relative_speed

            # Angle of attack
            cos_alpha = float(np.dot(normal, wind_dir))
            sin_alpha = math.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha))

            # Drag force (opposes motion, proportional to area facing wind)
            drag_magnitude = (
                0.5
                * self.settings.drag_coefficient
                * relative_speed * relative_speed
                * area
                * abs(cos_alpha)
            )
            drag_force = wind_dir * drag_magnitude

            # Lift force (perpendicular to wind, based on angle of attack)
            if sin_alpha > 1e-8:
                lift_direction = np.cross(np.cross(wind_dir, normal), wind_dir)
                lift_len = np.linalg.norm(lift_direction)
                if lift_len > 1e-8:
                    lift_direction /= lift_len
                    lift_magnitude = (
                        0.5
                        * self.settings.lift_coefficient
                        * relative_speed * relative_speed
                        * area
                        * sin_alpha * cos_alpha  # Lift peaks at 45 degrees
                    )
                    lift_force = lift_direction * lift_magnitude
                else:
                    lift_force = np.zeros(3, dtype=np.float32)
            else:
                lift_force = np.zeros(3, dtype=np.float32)

            # Total force on triangle
            total_force = (drag_force + lift_force) / 3.0  # Distribute to vertices

            # Apply to vertices with influence weighting
            for p, idx in [(p0, tri.p0), (p1, tri.p1), (p2, tri.p2)]:
                if p.inv_mass > 0:
                    influence = 1.0
                    if self._vertex_influence is not None and idx < len(
                        self._vertex_influence
                    ):
                        influence = self._vertex_influence[idx]

                    p.acceleration += total_force * p.inv_mass * influence

    def _get_wind_at_position(
        self,
        position: NDArray[np.float32],
        base_wind: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Get wind velocity at a position, including turbulence.

        Args:
            position: World position
            base_wind: Base wind velocity

        Returns:
            Wind velocity including turbulence
        """
        if self.settings.turbulence_strength < 1e-6:
            return base_wind

        # Sample turbulence noise
        turbulence = self._sample_turbulence(position)

        return base_wind + turbulence * self.settings.turbulence_strength

    def _sample_turbulence(
        self,
        position: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Sample 3D turbulence noise at a position.

        Uses simplified fractal noise (fBm).

        Args:
            position: Sampling position

        Returns:
            3D turbulence vector
        """
        result = np.zeros(3, dtype=np.float32)
        amplitude = 1.0
        frequency = self.settings.turbulence_frequency
        total_amplitude = 0.0

        for octave in range(self.settings.turbulence_octaves):
            # Sample noise for each axis
            sample_x = position * frequency + np.array(
                [self._time, 0.0, 0.0]
            )
            sample_y = position * frequency + np.array(
                [0.0, self._time, 0.0]
            )
            sample_z = position * frequency + np.array(
                [0.0, 0.0, self._time]
            )

            # Simplified noise using sin waves
            noise_x = self._simple_noise(sample_x, self._turbulence_seed)
            noise_y = self._simple_noise(sample_y, self._turbulence_seed + 1)
            noise_z = self._simple_noise(sample_z, self._turbulence_seed + 2)

            result += np.array(
                [noise_x, noise_y, noise_z], dtype=np.float32
            ) * amplitude

            total_amplitude += amplitude
            amplitude *= 0.5
            frequency *= 2.0

        if total_amplitude > 0:
            result /= total_amplitude

        return result

    def _simple_noise(
        self,
        position: NDArray[np.float32],
        seed: int,
    ) -> float:
        """
        Simple pseudo-random noise function.

        Not production quality - replace with proper Perlin/Simplex noise.

        Args:
            position: Sample position
            seed: Random seed

        Returns:
            Noise value in [-1, 1]
        """
        # Hash-based pseudo-random
        h = seed
        for i, v in enumerate(position):
            h ^= int(v * 1000) * (73856093 if i == 0 else 19349663 if i == 1 else 83492791)
            h = (h * 16777619) & 0xFFFFFFFF

        # Convert to float in [-1, 1]
        return (h / 0x7FFFFFFF) - 1.0


@dataclass
class DirectionalWind:
    """
    Simple directional wind source.

    Provides constant wind in a given direction.
    """

    direction: NDArray[np.float32]
    strength: float = 1.0

    def get_velocity(
        self,
        position: NDArray[np.float32],
        time: float,
    ) -> NDArray[np.float32]:
        """Get wind velocity at position."""
        return self.direction * self.strength


@dataclass
class PointWind:
    """
    Point wind source (like a fan).

    Wind radiates outward from a point.
    """

    position: NDArray[np.float32]
    strength: float = 1.0
    radius: float = 5.0
    falloff: float = 2.0  # Quadratic falloff

    def get_velocity(
        self,
        sample_position: NDArray[np.float32],
        time: float,
    ) -> NDArray[np.float32]:
        """Get wind velocity at position."""
        delta = sample_position - self.position
        distance = float(np.linalg.norm(delta))

        if distance < 1e-8 or distance > self.radius:
            return np.zeros(3, dtype=np.float32)

        # Direction from source to sample
        direction = delta / distance

        # Falloff with distance
        attenuation = math.pow(
            1.0 - distance / self.radius, self.falloff
        )

        return direction * self.strength * attenuation


@dataclass
class VortexWind:
    """
    Vortex wind source (rotating wind).

    Creates a swirling wind pattern around an axis.
    """

    center: NDArray[np.float32]
    axis: NDArray[np.float32]
    strength: float = 1.0
    radius: float = 5.0
    angular_velocity: float = 1.0  # radians per second

    def get_velocity(
        self,
        position: NDArray[np.float32],
        time: float,
    ) -> NDArray[np.float32]:
        """Get wind velocity at position."""
        # Vector from center to position
        to_pos = position - self.center

        # Project onto plane perpendicular to axis
        axis_norm = self.axis / np.linalg.norm(self.axis)
        along_axis = np.dot(to_pos, axis_norm) * axis_norm
        radial = to_pos - along_axis

        distance = float(np.linalg.norm(radial))

        if distance < 1e-8 or distance > self.radius:
            return np.zeros(3, dtype=np.float32)

        # Tangent direction (perpendicular to both axis and radial)
        tangent = np.cross(axis_norm, radial)
        tangent_len = np.linalg.norm(tangent)

        if tangent_len < 1e-8:
            return np.zeros(3, dtype=np.float32)

        tangent /= tangent_len

        # Velocity increases toward center, then drops off
        # Rankine vortex profile
        core_radius = self.radius * 0.2
        if distance < core_radius:
            speed = self.angular_velocity * distance
        else:
            speed = self.angular_velocity * core_radius * core_radius / distance

        return tangent * speed * self.strength


class WindSystem:
    """
    Wind system managing multiple wind sources.

    Combines multiple wind sources and applies them to cloth.
    """

    def __init__(self) -> None:
        """Initialize the wind system."""
        self._wind_force = WindForce()
        self._directional_winds: List[DirectionalWind] = []
        self._point_winds: List[PointWind] = []
        self._vortex_winds: List[VortexWind] = []
        self._time: float = 0.0

    def add_directional_wind(self, wind: DirectionalWind) -> None:
        """Add a directional wind source."""
        self._directional_winds.append(wind)

    def add_point_wind(self, wind: PointWind) -> None:
        """Add a point wind source."""
        self._point_winds.append(wind)

    def add_vortex_wind(self, wind: VortexWind) -> None:
        """Add a vortex wind source."""
        self._vortex_winds.append(wind)

    def clear_winds(self) -> None:
        """Remove all wind sources."""
        self._directional_winds.clear()
        self._point_winds.clear()
        self._vortex_winds.clear()

    def get_combined_wind(
        self,
        position: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Get combined wind velocity at a position.

        Args:
            position: World position

        Returns:
            Total wind velocity from all sources
        """
        total = np.zeros(3, dtype=np.float32)

        for wind in self._directional_winds:
            total += wind.get_velocity(position, self._time)

        for wind in self._point_winds:
            total += wind.get_velocity(position, self._time)

        for wind in self._vortex_winds:
            total += wind.get_velocity(position, self._time)

        return total

    def apply_to_mesh(
        self,
        mesh: ClothMesh,
        dt: float,
    ) -> None:
        """
        Apply wind forces to a cloth mesh.

        Args:
            mesh: The cloth mesh
            dt: Timestep
        """
        self._time += dt

        # If we have custom wind sources, compute combined wind
        if self._directional_winds or self._point_winds or self._vortex_winds:
            self._apply_combined_winds(mesh, dt)
        else:
            # Use default wind force
            self._wind_force.compute_wind_force(mesh, dt)

    def _apply_combined_winds(
        self,
        mesh: ClothMesh,
        dt: float,
    ) -> None:
        """Apply combined wind from all sources to mesh."""
        particles = mesh.particles

        for tri in mesh.triangles:
            p0 = particles[tri.p0]
            p1 = particles[tri.p1]
            p2 = particles[tri.p2]

            if p0.inv_mass == 0 and p1.inv_mass == 0 and p2.inv_mass == 0:
                continue

            # Triangle center
            center = (p0.position + p1.position + p2.position) / 3.0

            # Get combined wind at center
            wind_velocity = self.get_combined_wind(center)
            wind_speed = float(np.linalg.norm(wind_velocity))

            if wind_speed < 1e-8:
                continue

            # Triangle normal and area
            edge1 = p1.position - p0.position
            edge2 = p2.position - p0.position
            cross = np.cross(edge1, edge2)
            area_x2 = float(np.linalg.norm(cross))

            if area_x2 < 1e-8:
                continue

            normal = cross / area_x2
            area = area_x2 * 0.5

            # Force based on wind dot normal
            wind_dir = wind_velocity / wind_speed
            facing = abs(float(np.dot(normal, wind_dir)))

            # Simple drag force
            force_magnitude = (
                0.5
                * self._wind_force.settings.drag_coefficient
                * wind_speed * wind_speed
                * area
                * facing
            )

            force = wind_dir * force_magnitude / 3.0

            # Apply to vertices
            for p in [p0, p1, p2]:
                if p.inv_mass > 0:
                    p.acceleration += force * p.inv_mass
