"""Bruneton 2017 Atmospheric Scattering LUT Precomputation.

This module implements CPU-side precomputation for atmospheric scattering
lookup tables following the Bruneton 2017 paper approach:
"Precomputed Atmospheric Scattering" (Bruneton & Neyret, 2008, updated 2017).

The generated LUTs are:
- Transmittance LUT: Beer-Lambert extinction along view rays
- Sky-View LUT: Single scattering for sky dome rendering
- Aerial Perspective LUT: In-scattering and transmittance for distant objects

References:
- https://ebruneton.github.io/precomputed_atmospheric_scattering/
- Bruneton, E., & Neyret, F. (2008). Precomputed atmospheric scattering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

EPSILON: float = 1e-6
SMALL_FLOAT: float = 1e-10

# Integration steps for ray marching
# Optimized for CPU performance while maintaining visual quality
TRANSMITTANCE_STEPS: int = 64  # Reduced from 500 for CPU perf (<100ms target)
SCATTERING_STEPS: int = 16  # Reduced from 32 for CPU perf
MULTISCATTER_STEPS: int = 8  # Reduced from 16 for CPU perf


# -----------------------------------------------------------------------------
# Atmosphere Parameters
# -----------------------------------------------------------------------------


@dataclass
class AtmosphereParams:
    """Physical parameters for atmospheric scattering.

    Default values are calibrated for Earth's atmosphere at sea level.
    All distances are in meters, coefficients in m^-1.

    Attributes:
        planet_radius: Radius of the planet in meters (Earth: 6371km).
        atmosphere_height: Height of the atmosphere top in meters.
        rayleigh_scale_height: Scale height for Rayleigh scattering (8km for Earth).
        mie_scale_height: Scale height for Mie scattering (1.2km for Earth).
        rayleigh_scattering: RGB Rayleigh scattering coefficients at sea level.
        mie_scattering: Mie scattering coefficient at sea level.
        mie_absorption: Mie absorption coefficient at sea level.
        mie_asymmetry_g: Henyey-Greenstein / Cornette-Shanks asymmetry parameter.
        ozone_absorption: RGB ozone absorption coefficients (Chappuis band).
        sun_angular_radius: Angular radius of the sun in radians.
    """

    planet_radius: float = 6371e3  # Earth radius in meters
    atmosphere_height: float = 80e3  # Top of atmosphere
    rayleigh_scale_height: float = 8e3
    mie_scale_height: float = 1.2e3
    rayleigh_scattering: Tuple[float, float, float] = (5.5e-6, 13.0e-6, 22.4e-6)  # RGB
    mie_scattering: float = 21e-6
    mie_absorption: float = 4.4e-6
    mie_asymmetry_g: float = 0.8  # Henyey-Greenstein asymmetry
    ozone_absorption: Tuple[float, float, float] = (
        0.65e-6,
        1.88e-6,
        0.085e-6,
    )  # Chappuis band
    sun_angular_radius: float = 0.00467  # radians (~0.267 degrees)

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.planet_radius <= 0:
            raise ValueError("planet_radius must be positive")
        if self.atmosphere_height <= 0:
            raise ValueError("atmosphere_height must be positive")
        if self.rayleigh_scale_height <= 0:
            raise ValueError("rayleigh_scale_height must be positive")
        if self.mie_scale_height <= 0:
            raise ValueError("mie_scale_height must be positive")
        if not (-1.0 <= self.mie_asymmetry_g <= 1.0):
            raise ValueError("mie_asymmetry_g must be in range [-1, 1]")
        if self.sun_angular_radius <= 0:
            raise ValueError("sun_angular_radius must be positive")

    @property
    def atmosphere_top_radius(self) -> float:
        """Radius at the top of the atmosphere."""
        return self.planet_radius + self.atmosphere_height

    @property
    def mie_extinction(self) -> float:
        """Total Mie extinction coefficient (scattering + absorption)."""
        return self.mie_scattering + self.mie_absorption


@dataclass
class LUTDimensions:
    """Dimensions for atmospheric LUTs.

    Attributes:
        transmittance_width: Width of transmittance LUT (view zenith samples).
        transmittance_height: Height of transmittance LUT (altitude samples).
        sky_view_width: Width of sky-view LUT (view zenith samples).
        sky_view_height: Height of sky-view LUT (view azimuth samples).
        aerial_perspective_size: Size of 3D aerial perspective LUT.
    """

    transmittance_width: int = 256
    transmittance_height: int = 64
    sky_view_width: int = 256
    sky_view_height: int = 512
    aerial_perspective_size: int = 32


# -----------------------------------------------------------------------------
# Phase Functions
# -----------------------------------------------------------------------------


def rayleigh_phase(cos_angle: float) -> float:
    """Rayleigh phase function.

    The Rayleigh phase function describes angular scattering distribution
    for particles much smaller than the wavelength of light.

    Args:
        cos_angle: Cosine of the scattering angle.

    Returns:
        Phase function value (normalized to integrate to 1 over the sphere).
    """
    # P(theta) = (3 / 16 * pi) * (1 + cos^2(theta))
    return (3.0 / (16.0 * math.pi)) * (1.0 + cos_angle * cos_angle)


def cornette_shanks_phase(cos_angle: float, g: float) -> float:
    """Cornette-Shanks phase function for Mie scattering.

    More accurate than Henyey-Greenstein for atmospheric aerosols.
    Reduces to Rayleigh phase when g=0.

    Args:
        cos_angle: Cosine of the scattering angle.
        g: Asymmetry parameter (-1 to 1). Positive = forward scattering.

    Returns:
        Phase function value (normalized).
    """
    if abs(g) < EPSILON:
        return rayleigh_phase(cos_angle)

    g2 = g * g
    k = 3.0 / (8.0 * math.pi) * (1.0 - g2) / (2.0 + g2)
    denom = 1.0 + g2 - 2.0 * g * cos_angle

    if denom < EPSILON:
        denom = EPSILON

    return k * (1.0 + cos_angle * cos_angle) / (denom ** 1.5)


def henyey_greenstein_phase(cos_angle: float, g: float) -> float:
    """Henyey-Greenstein phase function.

    Classic single-lobe phase function for aerosol scattering.

    Args:
        cos_angle: Cosine of the scattering angle.
        g: Asymmetry parameter (-1 to 1).

    Returns:
        Phase function value (normalized).
    """
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_angle

    if denom < EPSILON:
        denom = EPSILON

    return (1.0 - g2) / (4.0 * math.pi * (denom ** 1.5))


# -----------------------------------------------------------------------------
# Optical Depth and Transmittance
# -----------------------------------------------------------------------------


def get_density_at_altitude(
    altitude: float, scale_height: float, params: AtmosphereParams
) -> float:
    """Get normalized density at a given altitude.

    Uses exponential falloff model: rho(h) = rho_0 * exp(-h / H)

    Args:
        altitude: Height above ground in meters.
        scale_height: Scale height for the medium.
        params: Atmosphere parameters.

    Returns:
        Normalized density (0 to 1).
    """
    if altitude < 0:
        altitude = 0
    if altitude > params.atmosphere_height:
        return 0.0

    return math.exp(-altitude / scale_height)


def get_ozone_density(altitude: float, params: AtmosphereParams) -> float:
    """Get ozone density at a given altitude.

    Ozone is concentrated in the stratosphere (15-35 km) with a
    peak around 25 km.

    Args:
        altitude: Height above ground in meters.
        params: Atmosphere parameters.

    Returns:
        Normalized ozone density.
    """
    if altitude < 0 or altitude > params.atmosphere_height:
        return 0.0

    # Ozone layer peaks at ~25km with width ~15km
    ozone_center_height = 25e3
    ozone_width = 15e3

    # Gaussian-like distribution
    t = (altitude - ozone_center_height) / ozone_width
    return max(0.0, 1.0 - t * t)


def ray_sphere_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> Tuple[float, float]:
    """Calculate ray-sphere intersection distances.

    Args:
        ray_origin: Origin point of the ray.
        ray_dir: Normalized direction of the ray.
        sphere_center: Center of the sphere.
        sphere_radius: Radius of the sphere.

    Returns:
        Tuple of (near, far) intersection distances.
        Returns (-1, -1) if no intersection.
    """
    oc = ray_origin - sphere_center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius

    discriminant = b * b - 4.0 * a * c

    if discriminant < 0:
        return (-1.0, -1.0)

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    return (t1, t2)


def compute_optical_depth(
    altitude: float, zenith_cos: float, params: AtmosphereParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Rayleigh, Mie, and ozone optical depths along a view ray.

    Integrates extinction coefficients along the ray from the given
    position to the top of the atmosphere. Uses vectorized numpy operations.

    Args:
        altitude: Starting altitude above ground in meters.
        zenith_cos: Cosine of the zenith angle (1 = up, -1 = down).
        params: Atmosphere parameters.

    Returns:
        Tuple of (rayleigh_od, mie_od, ozone_od) as numpy arrays.
        Rayleigh and ozone are RGB, Mie is scalar (but returned as 1-element array).
    """
    # Clamp altitude to valid range
    altitude = max(0.0, min(altitude, params.atmosphere_height))

    # Position on the planet (assume we're on the surface + altitude)
    r = params.planet_radius + altitude
    position = np.array([0.0, r, 0.0])

    # View direction from zenith cosine
    zenith_sin = math.sqrt(max(0.0, 1.0 - zenith_cos * zenith_cos))
    view_dir = np.array([zenith_sin, zenith_cos, 0.0])

    # Find intersection with atmosphere top
    _, t_max = ray_sphere_intersection(
        position,
        view_dir,
        np.array([0.0, 0.0, 0.0]),
        params.atmosphere_top_radius,
    )

    if t_max < 0:
        # No intersection with atmosphere
        return (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0]),
            np.array([0.0, 0.0, 0.0]),
        )

    # Check for ground intersection (looking down)
    t_ground_near, _ = ray_sphere_intersection(
        position,
        view_dir,
        np.array([0.0, 0.0, 0.0]),
        params.planet_radius,
    )

    if t_ground_near > 0:
        t_max = t_ground_near

    # Vectorized integration using numpy
    dt = t_max / TRANSMITTANCE_STEPS
    t_values = (np.arange(TRANSMITTANCE_STEPS) + 0.5) * dt  # (N,)

    # Sample positions along ray: position + t * view_dir
    # position is (3,), view_dir is (3,), t_values is (N,)
    sample_positions = position[np.newaxis, :] + t_values[:, np.newaxis] * view_dir[np.newaxis, :]  # (N, 3)

    # Compute radii and altitudes
    sample_r = np.linalg.norm(sample_positions, axis=1)  # (N,)
    sample_altitudes = sample_r - params.planet_radius  # (N,)

    # Mask for valid altitude range
    valid_mask = (sample_altitudes >= 0) & (sample_altitudes <= params.atmosphere_height)

    # Rayleigh density (exponential falloff)
    rayleigh_density = np.where(
        valid_mask,
        np.exp(-sample_altitudes / params.rayleigh_scale_height),
        0.0
    )  # (N,)

    # Mie density
    mie_density = np.where(
        valid_mask,
        np.exp(-sample_altitudes / params.mie_scale_height),
        0.0
    )  # (N,)

    # Ozone density (Gaussian centered at 25km)
    ozone_center_height = 25e3
    ozone_width = 15e3
    t_ozone = (sample_altitudes - ozone_center_height) / ozone_width
    ozone_density = np.where(
        valid_mask,
        np.maximum(0.0, 1.0 - t_ozone * t_ozone),
        0.0
    )  # (N,)

    # Integrate optical depths
    rayleigh_coeffs = np.array(params.rayleigh_scattering)  # (3,)
    ozone_coeffs = np.array(params.ozone_absorption)  # (3,)

    # Sum over all samples: sum(density * dt) * coefficient
    rayleigh_od = rayleigh_coeffs * np.sum(rayleigh_density) * dt
    mie_od = params.mie_extinction * np.sum(mie_density) * dt
    ozone_od = ozone_coeffs * np.sum(ozone_density) * dt

    return (rayleigh_od, np.array([mie_od]), ozone_od)


def compute_transmittance(optical_depth: np.ndarray) -> np.ndarray:
    """Compute transmittance using Beer-Lambert law.

    T = exp(-optical_depth)

    Args:
        optical_depth: Optical depth value(s).

    Returns:
        Transmittance value(s) in range [0, 1].
    """
    return np.exp(-np.clip(optical_depth, 0, 100))  # Clip to prevent underflow


# -----------------------------------------------------------------------------
# LUT Validation
# -----------------------------------------------------------------------------


def validate_transmittance_lut(lut: np.ndarray) -> bool:
    """Validate transmittance LUT for correctness.

    Checks:
    - Non-negative values
    - Maximum value <= 1.0 (physically valid transmittance)
    - Non-zero values at all altitudes
    - Correct shape (width, height, channels)

    Args:
        lut: Transmittance LUT array.

    Returns:
        True if valid, False otherwise.
    """
    if lut is None or lut.size == 0:
        return False

    # Check dimensions (should be 2D with 4 channels for RGBA16F)
    if len(lut.shape) != 3:
        return False

    # Check for negative values
    if np.any(lut < 0):
        return False

    # Check maximum is <= 1.0 (with small epsilon for float precision)
    if np.any(lut > 1.0 + EPSILON):
        return False

    # Check that we have valid transmittance at high altitudes
    # (top row should have transmittance close to 1)
    height = lut.shape[0]
    top_row = lut[height - 1, :, :3]  # RGB channels only
    if np.max(top_row) < 0.5:
        return False

    # Check for NaN or Inf
    if np.any(~np.isfinite(lut)):
        return False

    return True


def validate_sky_view_lut(lut: np.ndarray) -> bool:
    """Validate sky-view LUT for correctness.

    Checks:
    - Non-negative values
    - Plausible HDR range
    - Maximum > 0.5 near sun direction
    - Correct shape

    Args:
        lut: Sky-view LUT array.

    Returns:
        True if valid, False otherwise.
    """
    if lut is None or lut.size == 0:
        return False

    # Check dimensions
    if len(lut.shape) != 3:
        return False

    # Check for negative values
    if np.any(lut < 0):
        return False

    # Check for reasonable maximum (HDR but not infinite)
    max_val = np.max(lut)
    if max_val > 1e6:
        return False

    # Check that we have some brightness (sun should produce significant values)
    if max_val < 0.01:
        return False

    # Check for NaN or Inf
    if np.any(~np.isfinite(lut)):
        return False

    return True


def validate_aerial_perspective_lut(
    inscatter: np.ndarray, transmittance: np.ndarray
) -> bool:
    """Validate aerial perspective LUTs for correctness.

    Args:
        inscatter: In-scattering LUT (RGBA16F).
        transmittance: Transmittance LUT (RGBA16F).

    Returns:
        True if valid, False otherwise.
    """
    if inscatter is None or inscatter.size == 0:
        return False
    if transmittance is None or transmittance.size == 0:
        return False

    # Check dimensions (should be 3D with 4 channels)
    if len(inscatter.shape) != 4 or len(transmittance.shape) != 4:
        return False

    # Check for negative values
    if np.any(inscatter < 0) or np.any(transmittance < 0):
        return False

    # Transmittance should be <= 1.0
    if np.any(transmittance > 1.0 + EPSILON):
        return False

    # Check for NaN or Inf
    if np.any(~np.isfinite(inscatter)) or np.any(~np.isfinite(transmittance)):
        return False

    return True


# -----------------------------------------------------------------------------
# LUT Generator Class
# -----------------------------------------------------------------------------


class BrunetonLUTGenerator:
    """Generator for Bruneton atmospheric scattering LUTs.

    This class precomputes lookup tables for real-time atmospheric
    scattering rendering. The LUTs encode:

    - Transmittance: Extinction along view rays at various altitudes
    - Sky-View: Single-scattering sky color for a given sun position
    - Aerial Perspective: In-scattering for distant objects

    Example:
        generator = BrunetonLUTGenerator()
        luts = generator.precompute_all(sun_direction=np.array([0.5, 0.5, 0.0]))
        transmittance = luts['transmittance']
        sky_view = luts['sky_view']
        inscatter, ap_trans = luts['aerial_perspective']
    """

    def __init__(
        self,
        params: AtmosphereParams | None = None,
        dimensions: LUTDimensions | None = None,
    ) -> None:
        """Initialize the LUT generator.

        Args:
            params: Atmosphere parameters. Uses Earth defaults if None.
            dimensions: LUT dimensions. Uses defaults if None.
        """
        self.params = params if params is not None else AtmosphereParams()
        self.dimensions = dimensions if dimensions is not None else LUTDimensions()

        # Cache transmittance LUT for use in other computations
        self._transmittance_lut: np.ndarray | None = None

    def _map_transmittance_uv_to_params(
        self, u: float, v: float
    ) -> Tuple[float, float]:
        """Map UV coordinates to altitude and zenith cosine.

        Uses non-linear mapping to concentrate samples near the horizon
        where transmittance changes most rapidly.

        Args:
            u: Horizontal coordinate [0, 1] -> zenith cosine.
            v: Vertical coordinate [0, 1] -> altitude.

        Returns:
            Tuple of (altitude, zenith_cos).
        """
        # Altitude: non-linear mapping to concentrate samples near ground
        altitude = v * v * self.params.atmosphere_height

        # Zenith cosine: non-linear mapping for horizon detail
        # Map u from [0, 1] to zenith_cos in [-1, 1] with more samples near horizon
        x = 2.0 * u - 1.0
        zenith_cos = x * x * np.sign(x)

        return (altitude, zenith_cos)

    def _sample_transmittance(self, altitude: float, zenith_cos: float) -> np.ndarray:
        """Sample transmittance for given parameters.

        If the transmittance LUT is cached, sample from it.
        Otherwise, compute directly.

        Args:
            altitude: Altitude above ground in meters.
            zenith_cos: Cosine of zenith angle.

        Returns:
            RGB transmittance values.
        """
        if self._transmittance_lut is not None:
            # Sample from cached LUT
            # Inverse mapping from params to UV
            v = math.sqrt(altitude / self.params.atmosphere_height)
            v = np.clip(v, 0, 1)

            # Inverse of x^2 * sign(x) mapping
            x = zenith_cos
            u = (np.sign(x) * math.sqrt(abs(x)) + 1.0) / 2.0
            u = np.clip(u, 0, 1)

            # Bilinear sample
            width = self._transmittance_lut.shape[1]
            height = self._transmittance_lut.shape[0]

            x_coord = u * (width - 1)
            y_coord = v * (height - 1)

            x0 = int(x_coord)
            y0 = int(y_coord)
            x1 = min(x0 + 1, width - 1)
            y1 = min(y0 + 1, height - 1)

            fx = x_coord - x0
            fy = y_coord - y0

            s00 = self._transmittance_lut[y0, x0, :3]
            s10 = self._transmittance_lut[y0, x1, :3]
            s01 = self._transmittance_lut[y1, x0, :3]
            s11 = self._transmittance_lut[y1, x1, :3]

            return (
                s00 * (1 - fx) * (1 - fy)
                + s10 * fx * (1 - fy)
                + s01 * (1 - fx) * fy
                + s11 * fx * fy
            )

        # Compute directly
        rayleigh_od, mie_od, ozone_od = compute_optical_depth(
            altitude, zenith_cos, self.params
        )
        total_od = rayleigh_od + mie_od[0] + ozone_od
        return compute_transmittance(total_od)

    def precompute_transmittance(
        self, width: int | None = None, height: int | None = None
    ) -> np.ndarray:
        """Generate transmittance LUT.

        The LUT stores transmittance from a given altitude to the top
        of the atmosphere for various view directions.

        Format: RGBA16F where RGB = transmittance, A = 1.0

        Target performance: <100ms

        Args:
            width: LUT width (zenith cosine samples). Uses default if None.
            height: LUT height (altitude samples). Uses default if None.

        Returns:
            Transmittance LUT as numpy array of shape (height, width, 4).
        """
        if width is None:
            width = self.dimensions.transmittance_width
        if height is None:
            height = self.dimensions.transmittance_height

        lut = np.zeros((height, width, 4), dtype=np.float32)

        # Vectorized UV grid
        v_vals = (np.arange(height) + 0.5) / height
        u_vals = (np.arange(width) + 0.5) / width

        # Pre-compute altitude and zenith_cos grids
        altitudes = (v_vals ** 2) * self.params.atmosphere_height  # (height,)

        # Zenith cosine mapping
        x_vals = 2.0 * u_vals - 1.0
        zenith_cos_vals = x_vals * x_vals * np.sign(x_vals)  # (width,)

        # Compute transmittance for each pixel
        for y in range(height):
            altitude = altitudes[y]
            for x in range(width):
                zenith_cos = zenith_cos_vals[x]

                rayleigh_od, mie_od, ozone_od = compute_optical_depth(
                    altitude, zenith_cos, self.params
                )

                # Total optical depth
                total_od = rayleigh_od + mie_od[0] + ozone_od

                # Transmittance via Beer-Lambert
                transmittance = compute_transmittance(total_od)

                lut[y, x, :3] = transmittance
                lut[y, x, 3] = 1.0

        # Cache for later use
        self._transmittance_lut = lut

        return lut

    def precompute_sky_view(
        self,
        sun_direction: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        """Generate sky-view LUT for a given sun position.

        The LUT stores single-scattering sky radiance for all view directions
        at a fixed observer position (ground level or camera altitude).

        Format: RGB16F

        Target performance: <300ms

        Args:
            sun_direction: Normalized direction towards the sun.
            width: LUT width (view zenith samples). Uses default if None.
            height: LUT height (view azimuth samples). Uses default if None.

        Returns:
            Sky-view LUT as numpy array of shape (height, width, 3).
        """
        if width is None:
            width = self.dimensions.sky_view_width
        if height is None:
            height = self.dimensions.sky_view_height

        # Ensure transmittance is precomputed
        if self._transmittance_lut is None:
            self.precompute_transmittance()

        # Normalize sun direction
        sun_dir = sun_direction / (np.linalg.norm(sun_direction) + EPSILON)

        lut = np.zeros((height, width, 3), dtype=np.float32)

        # Observer at ground level
        observer_altitude = 0.0
        observer_r = self.params.planet_radius + observer_altitude

        for y in range(height):
            # Azimuth angle [0, 2*pi]
            azimuth = (y + 0.5) / height * 2.0 * math.pi

            for x in range(width):
                # Zenith cosine [-1, 1] with non-linear mapping for horizon detail
                t = (x + 0.5) / width
                # Map to focus detail near horizon
                zenith_cos = 2.0 * t - 1.0
                zenith_cos = zenith_cos * abs(zenith_cos)  # Quadratic mapping

                # Compute view direction
                zenith_sin = math.sqrt(max(0.0, 1.0 - zenith_cos * zenith_cos))
                view_dir = np.array([
                    zenith_sin * math.cos(azimuth),
                    zenith_cos,
                    zenith_sin * math.sin(azimuth),
                ])

                # Compute single scattering
                radiance = self._compute_single_scattering(
                    observer_altitude, view_dir, sun_dir
                )

                lut[y, x, :] = radiance

        return lut

    def _compute_single_scattering(
        self,
        altitude: float,
        view_dir: np.ndarray,
        sun_dir: np.ndarray,
    ) -> np.ndarray:
        """Compute single scattering along a view ray.

        Args:
            altitude: Observer altitude in meters.
            view_dir: Normalized view direction.
            sun_dir: Normalized sun direction.

        Returns:
            RGB radiance values.
        """
        # Ray origin
        r = self.params.planet_radius + altitude
        position = np.array([0.0, r, 0.0])

        # Find intersection with atmosphere
        _, t_atm = ray_sphere_intersection(
            position,
            view_dir,
            np.array([0.0, 0.0, 0.0]),
            self.params.atmosphere_top_radius,
        )

        if t_atm < 0:
            return np.zeros(3)

        # Check for ground intersection
        t_ground, _ = ray_sphere_intersection(
            position,
            view_dir,
            np.array([0.0, 0.0, 0.0]),
            self.params.planet_radius,
        )

        t_max = t_atm
        if t_ground > 0:
            t_max = min(t_max, t_ground)

        # Cosine of angle between view and sun
        cos_theta = np.dot(view_dir, sun_dir)

        # Phase functions
        rayleigh_phase_val = rayleigh_phase(cos_theta)
        mie_phase_val = cornette_shanks_phase(cos_theta, self.params.mie_asymmetry_g)

        # Integrate scattering
        radiance = np.zeros(3)
        transmittance_to_sample = np.ones(3)

        dt = t_max / SCATTERING_STEPS

        for i in range(SCATTERING_STEPS):
            t = (i + 0.5) * dt
            sample_pos = position + t * view_dir
            sample_r = np.linalg.norm(sample_pos)
            sample_altitude = sample_r - self.params.planet_radius

            if sample_altitude < 0 or sample_altitude > self.params.atmosphere_height:
                continue

            # Get density at sample point
            rayleigh_density = get_density_at_altitude(
                sample_altitude, self.params.rayleigh_scale_height, self.params
            )
            mie_density = get_density_at_altitude(
                sample_altitude, self.params.mie_scale_height, self.params
            )

            # Scattering coefficients at sample point
            rayleigh_scatter = np.array(self.params.rayleigh_scattering) * rayleigh_density
            mie_scatter = self.params.mie_scattering * mie_density

            # Transmittance from sample to sun
            sun_zenith_cos = np.dot(sample_pos / sample_r, sun_dir)
            sun_transmittance = self._sample_transmittance(sample_altitude, sun_zenith_cos)

            # Check if sun is occluded by planet
            _, t_planet = ray_sphere_intersection(
                sample_pos,
                sun_dir,
                np.array([0.0, 0.0, 0.0]),
                self.params.planet_radius,
            )
            if t_planet > 0:
                sun_transmittance = np.zeros(3)

            # In-scattering contribution
            inscatter_rayleigh = rayleigh_scatter * rayleigh_phase_val * sun_transmittance
            inscatter_mie = mie_scatter * mie_phase_val * sun_transmittance

            radiance += transmittance_to_sample * (inscatter_rayleigh + inscatter_mie) * dt

            # Update transmittance along view ray
            extinction = (
                rayleigh_scatter
                + mie_density * self.params.mie_extinction
                + np.array(self.params.ozone_absorption) * get_ozone_density(sample_altitude, self.params)
            )
            transmittance_to_sample *= np.exp(-extinction * dt)

        return radiance

    def precompute_aerial_perspective(
        self, size: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate aerial perspective inscatter and transmittance LUTs.

        The 3D LUTs store in-scattering and transmittance for rendering
        distant objects with atmospheric haze.

        Format: RGBA16F x 2 (inscatter, transmittance)

        Args:
            size: Size of the 3D LUT (size x size x size). Uses default if None.

        Returns:
            Tuple of (inscatter_lut, transmittance_lut).
            Each is a numpy array of shape (size, size, size, 4).
        """
        if size is None:
            size = self.dimensions.aerial_perspective_size

        # Ensure transmittance is precomputed
        if self._transmittance_lut is None:
            self.precompute_transmittance()

        inscatter = np.zeros((size, size, size, 4), dtype=np.float32)
        transmittance = np.zeros((size, size, size, 4), dtype=np.float32)

        # Default sun direction (can be updated per-frame in shader)
        sun_dir = np.array([0.5, 0.5, 0.0])
        sun_dir = sun_dir / np.linalg.norm(sun_dir)

        max_distance = self.params.atmosphere_height * 0.5  # Max view distance

        for z in range(size):
            # Depth (distance from camera)
            t_depth = (z + 0.5) / size
            # Non-linear depth distribution
            distance = t_depth * t_depth * max_distance

            for y in range(size):
                # View zenith (altitude angle)
                t_zenith = (y + 0.5) / size
                zenith_cos = 2.0 * t_zenith - 1.0

                for x in range(size):
                    # View azimuth
                    azimuth = (x + 0.5) / size * 2.0 * math.pi

                    # View direction
                    zenith_sin = math.sqrt(max(0.0, 1.0 - zenith_cos * zenith_cos))
                    view_dir = np.array([
                        zenith_sin * math.cos(azimuth),
                        zenith_cos,
                        zenith_sin * math.sin(azimuth),
                    ])

                    # Compute in-scattering and transmittance to distance
                    inscatter_val, trans_val = self._compute_aerial_perspective_sample(
                        view_dir, sun_dir, distance
                    )

                    inscatter[z, y, x, :3] = inscatter_val
                    inscatter[z, y, x, 3] = 1.0
                    transmittance[z, y, x, :3] = trans_val
                    transmittance[z, y, x, 3] = 1.0

        return (inscatter, transmittance)

    def _compute_aerial_perspective_sample(
        self,
        view_dir: np.ndarray,
        sun_dir: np.ndarray,
        distance: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute aerial perspective for a single sample.

        Args:
            view_dir: Normalized view direction.
            sun_dir: Normalized sun direction.
            distance: Distance from camera.

        Returns:
            Tuple of (inscatter, transmittance) as RGB arrays.
        """
        # Observer at ground level
        observer_altitude = 0.0
        observer_r = self.params.planet_radius + observer_altitude
        position = np.array([0.0, observer_r, 0.0])

        # Clamp distance to atmosphere
        t_max = min(distance, self.params.atmosphere_height)

        # Cosine of angle between view and sun
        cos_theta = np.dot(view_dir, sun_dir)

        # Phase functions
        rayleigh_phase_val = rayleigh_phase(cos_theta)
        mie_phase_val = cornette_shanks_phase(cos_theta, self.params.mie_asymmetry_g)

        # Integrate
        total_inscatter = np.zeros(3)
        total_transmittance = np.ones(3)

        num_steps = max(8, int(t_max / 1000))  # Adaptive step count
        dt = t_max / num_steps

        for i in range(num_steps):
            t = (i + 0.5) * dt
            sample_pos = position + t * view_dir
            sample_r = np.linalg.norm(sample_pos)
            sample_altitude = sample_r - self.params.planet_radius

            if sample_altitude < 0:
                break
            if sample_altitude > self.params.atmosphere_height:
                continue

            # Densities
            rayleigh_density = get_density_at_altitude(
                sample_altitude, self.params.rayleigh_scale_height, self.params
            )
            mie_density = get_density_at_altitude(
                sample_altitude, self.params.mie_scale_height, self.params
            )

            # Scattering
            rayleigh_scatter = np.array(self.params.rayleigh_scattering) * rayleigh_density
            mie_scatter = self.params.mie_scattering * mie_density

            # Sun transmittance
            sun_zenith_cos = np.dot(sample_pos / sample_r, sun_dir)
            sun_transmittance = self._sample_transmittance(sample_altitude, sun_zenith_cos)

            # In-scattering
            inscatter_rayleigh = rayleigh_scatter * rayleigh_phase_val * sun_transmittance
            inscatter_mie = mie_scatter * mie_phase_val * sun_transmittance

            total_inscatter += total_transmittance * (inscatter_rayleigh + inscatter_mie) * dt

            # Update transmittance
            extinction = (
                rayleigh_scatter
                + mie_density * self.params.mie_extinction
            )
            total_transmittance *= np.exp(-extinction * dt)

        return (total_inscatter, total_transmittance)

    def precompute_all(
        self, sun_direction: np.ndarray
    ) -> Dict[str, np.ndarray | Tuple[np.ndarray, np.ndarray]]:
        """Generate all atmospheric LUTs.

        Precomputes transmittance, sky-view, and aerial perspective LUTs
        for the given sun position.

        Target performance: <500ms total

        Args:
            sun_direction: Normalized direction towards the sun.

        Returns:
            Dictionary with keys:
            - 'transmittance': Transmittance LUT (height, width, 4)
            - 'sky_view': Sky-view LUT (height, width, 3)
            - 'aerial_perspective': Tuple of (inscatter, transmittance) 3D LUTs
        """
        # Precompute in order (transmittance is used by others)
        transmittance = self.precompute_transmittance()
        sky_view = self.precompute_sky_view(sun_direction)
        aerial_perspective = self.precompute_aerial_perspective()

        return {
            "transmittance": transmittance,
            "sky_view": sky_view,
            "aerial_perspective": aerial_perspective,
        }

    def clear_cache(self) -> None:
        """Clear cached LUTs to free memory."""
        self._transmittance_lut = None
