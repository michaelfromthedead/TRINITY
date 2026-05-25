"""
PCG Noise Generation Utilities.

Provides various noise generation algorithms for procedural content generation:
- Perlin Noise: Gradient-based, smooth coherent noise
- Simplex Noise: Faster with fewer artifacts than Perlin
- Worley Noise: Cell/Voronoi-based distance noise
- Value Noise: Random values at grid points, interpolated
- Fractal Noise: Layered octaves of any base noise

All generators are deterministic given the same seed.
Uses Trinity Pattern with @seeded for deterministic generation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple


class NoiseType(Enum):
    """Types of noise generation algorithms."""

    PERLIN = auto()
    SIMPLEX = auto()
    WORLEY = auto()
    VALUE = auto()
    WHITE = auto()


@dataclass
class NoiseSettings:
    """Configuration for noise generation."""

    noise_type: NoiseType = NoiseType.PERLIN
    seed: int = 0
    frequency: float = 1.0
    octaves: int = 4
    lacunarity: float = 2.0
    persistence: float = 0.5
    amplitude: float = 1.0
    offset: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate settings after initialization."""
        if self.frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {self.frequency}")
        if self.octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {self.octaves}")
        if self.lacunarity <= 0:
            raise ValueError(f"lacunarity must be > 0, got {self.lacunarity}")
        if not 0 < self.persistence <= 1:
            raise ValueError(f"persistence must be in (0, 1], got {self.persistence}")
        if self.amplitude <= 0:
            raise ValueError(f"amplitude must be > 0, got {self.amplitude}")


class NoiseGenerator(ABC):
    """
    Abstract base class for noise generators.

    All implementations must be deterministic given the same seed.
    Output range is normalized to [-1, 1].
    """

    def __init__(self, seed: int = 0, settings: Optional[NoiseSettings] = None) -> None:
        """
        Initialize the noise generator.

        Args:
            seed: Random seed for deterministic generation
            settings: Noise configuration settings
        """
        self._seed = seed
        self._settings = settings or NoiseSettings(seed=seed)
        self._initialized = False
        self._initialize()

    @property
    def seed(self) -> int:
        """Get the generator seed."""
        return self._seed

    @property
    def settings(self) -> NoiseSettings:
        """Get the generator settings."""
        return self._settings

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize internal state for the generator."""
        pass

    @abstractmethod
    def sample(self, x: float, y: float) -> float:
        """
        Sample 2D noise at the given coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        pass

    def sample_3d(self, x: float, y: float, z: float) -> float:
        """
        Sample 3D noise at the given coordinates.

        Default implementation uses 2D noise with z as a perturbation.
        Subclasses should override for true 3D noise.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Default: perturb 2D coordinates with z
        perturbation = z * 0.7654321
        return self.sample(x + perturbation, y + perturbation * 0.5678)

    def _hash(self, *values: int) -> int:
        """Hash multiple integer values deterministically."""
        h = self._seed
        for v in values:
            h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2)
        return h & 0x7FFFFFFF


class PerlinNoise(NoiseGenerator):
    """
    Perlin noise generator using gradient-based interpolation.

    Classic coherent noise with smooth transitions between values.
    Output range: [-1, 1]
    """

    _PERMUTATION_SIZE = 256

    def __init__(self, seed: int = 0, settings: Optional[NoiseSettings] = None) -> None:
        self._permutation: List[int] = []
        self._gradients_2d: List[Tuple[float, float]] = []
        self._gradients_3d: List[Tuple[float, float, float]] = []
        super().__init__(seed, settings)

    def _initialize(self) -> None:
        """Generate permutation table and gradients."""
        self._generate_permutation()
        self._generate_gradients()
        self._initialized = True

    def _generate_permutation(self) -> None:
        """Generate a deterministic permutation table."""
        # Initialize with sequential values
        self._permutation = list(range(self._PERMUTATION_SIZE))

        # Fisher-Yates shuffle with deterministic random
        state = self._seed
        for i in range(self._PERMUTATION_SIZE - 1, 0, -1):
            # LCG random number generator
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            j = state % (i + 1)
            self._permutation[i], self._permutation[j] = (
                self._permutation[j],
                self._permutation[i],
            )

        # Double the permutation to avoid wrapping
        self._permutation = self._permutation + self._permutation

    def _generate_gradients(self) -> None:
        """Generate unit gradient vectors."""
        # 2D gradients - 8 directions
        self._gradients_2d = [
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.7071067811865476, 0.7071067811865476),
            (-0.7071067811865476, 0.7071067811865476),
            (0.7071067811865476, -0.7071067811865476),
            (-0.7071067811865476, -0.7071067811865476),
        ]

        # 3D gradients - 12 edge directions
        self._gradients_3d = [
            (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
            (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
            (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
        ]

    def _get_gradient_2d(self, ix: int, iy: int) -> Tuple[float, float]:
        """Get gradient at integer grid point."""
        idx = self._permutation[(ix + self._permutation[iy & 255]) & 255] & 7
        return self._gradients_2d[idx]

    def _get_gradient_3d(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        """Get gradient at integer 3D grid point."""
        idx = self._permutation[
            (ix + self._permutation[(iy + self._permutation[iz & 255]) & 255]) & 255
        ] % 12
        return self._gradients_3d[idx]

    def _dot_grid_gradient(
        self, ix: int, iy: int, x: float, y: float
    ) -> float:
        """Compute dot product between gradient and distance vector."""
        gradient = self._get_gradient_2d(ix, iy)
        dx = x - ix
        dy = y - iy
        return dx * gradient[0] + dy * gradient[1]

    def _dot_grid_gradient_3d(
        self, ix: int, iy: int, iz: int, x: float, y: float, z: float
    ) -> float:
        """Compute dot product for 3D gradient."""
        gradient = self._get_gradient_3d(ix, iy, iz)
        dx = x - ix
        dy = y - iy
        dz = z - iz
        return dx * gradient[0] + dy * gradient[1] + dz * gradient[2]

    @staticmethod
    def _interpolate(a: float, b: float, t: float) -> float:
        """
        Smoothstep interpolation between a and b.

        Uses 6t^5 - 15t^4 + 10t^3 (improved Perlin fade function).
        """
        # Improved fade function: 6t^5 - 15t^4 + 10t^3
        t = t * t * t * (t * (t * 6 - 15) + 10)
        return a + t * (b - a)

    def sample(self, x: float, y: float) -> float:
        """
        Sample 2D Perlin noise.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Apply frequency and offset
        x = x * self._settings.frequency + self._settings.offset[0]
        y = y * self._settings.frequency + self._settings.offset[1]

        # Grid cell coordinates (integer parts)
        xi = int(math.floor(x))
        yi = int(math.floor(y))

        # Wrapped grid coordinates for gradient lookup
        x0 = xi & 255
        y0 = yi & 255
        x1 = (x0 + 1) & 255
        y1 = (y0 + 1) & 255

        # Local coordinates within cell [0, 1]
        xf = x - xi
        yf = y - yi

        # Dot products at corners (use local coordinates for distance)
        n00 = self._dot_grid_gradient_local(x0, y0, xf, yf)
        n10 = self._dot_grid_gradient_local(x1, y0, xf - 1, yf)
        n01 = self._dot_grid_gradient_local(x0, y1, xf, yf - 1)
        n11 = self._dot_grid_gradient_local(x1, y1, xf - 1, yf - 1)

        # Interpolate using smoothstep
        u = self._fade(xf)
        v = self._fade(yf)

        nx0 = self._lerp(n00, n10, u)
        nx1 = self._lerp(n01, n11, u)
        value = self._lerp(nx0, nx1, v)

        # Apply amplitude
        return value * self._settings.amplitude

    def _dot_grid_gradient_local(
        self, ix: int, iy: int, dx: float, dy: float
    ) -> float:
        """Compute dot product between gradient and distance vector."""
        gradient = self._get_gradient_2d(ix, iy)
        return dx * gradient[0] + dy * gradient[1]

    @staticmethod
    def _fade(t: float) -> float:
        """Fade function: 6t^5 - 15t^4 + 10t^3."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + t * (b - a)

    def sample_3d(self, x: float, y: float, z: float) -> float:
        """
        Sample 3D Perlin noise.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Apply frequency
        x = x * self._settings.frequency + self._settings.offset[0]
        y = y * self._settings.frequency + self._settings.offset[1]
        z = z * self._settings.frequency

        # Grid cell coordinates (integer parts)
        xi = int(math.floor(x))
        yi = int(math.floor(y))
        zi = int(math.floor(z))

        # Wrapped grid coordinates for gradient lookup
        x0 = xi & 255
        y0 = yi & 255
        z0 = zi & 255
        x1 = (x0 + 1) & 255
        y1 = (y0 + 1) & 255
        z1 = (z0 + 1) & 255

        # Local coordinates within cell [0, 1]
        xf = x - xi
        yf = y - yi
        zf = z - zi

        # Dot products at 8 corners using local coordinates
        n000 = self._dot_grid_gradient_3d_local(x0, y0, z0, xf, yf, zf)
        n100 = self._dot_grid_gradient_3d_local(x1, y0, z0, xf - 1, yf, zf)
        n010 = self._dot_grid_gradient_3d_local(x0, y1, z0, xf, yf - 1, zf)
        n110 = self._dot_grid_gradient_3d_local(x1, y1, z0, xf - 1, yf - 1, zf)
        n001 = self._dot_grid_gradient_3d_local(x0, y0, z1, xf, yf, zf - 1)
        n101 = self._dot_grid_gradient_3d_local(x1, y0, z1, xf - 1, yf, zf - 1)
        n011 = self._dot_grid_gradient_3d_local(x0, y1, z1, xf, yf - 1, zf - 1)
        n111 = self._dot_grid_gradient_3d_local(x1, y1, z1, xf - 1, yf - 1, zf - 1)

        # Apply fade curve
        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        # Trilinear interpolation
        nx00 = self._lerp(n000, n100, u)
        nx10 = self._lerp(n010, n110, u)
        nx01 = self._lerp(n001, n101, u)
        nx11 = self._lerp(n011, n111, u)

        nxy0 = self._lerp(nx00, nx10, v)
        nxy1 = self._lerp(nx01, nx11, v)

        value = self._lerp(nxy0, nxy1, w)

        return value * self._settings.amplitude

    def _dot_grid_gradient_3d_local(
        self, ix: int, iy: int, iz: int, dx: float, dy: float, dz: float
    ) -> float:
        """Compute dot product for 3D gradient using local distances."""
        gradient = self._get_gradient_3d(ix, iy, iz)
        return dx * gradient[0] + dy * gradient[1] + dz * gradient[2]


class SimplexNoise(NoiseGenerator):
    """
    Simplex noise generator.

    Faster than Perlin with fewer directional artifacts.
    Uses a simplex grid instead of a square grid.
    Output range: [-1, 1]
    """

    # Skewing factors for 2D
    _F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    _G2 = (3.0 - math.sqrt(3.0)) / 6.0

    # Skewing factors for 3D
    _F3 = 1.0 / 3.0
    _G3 = 1.0 / 6.0

    def __init__(self, seed: int = 0, settings: Optional[NoiseSettings] = None) -> None:
        self._perm: List[int] = []
        self._perm_mod12: List[int] = []
        super().__init__(seed, settings)

    # Gradient vectors for 2D
    _GRAD2 = [
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (1, 0), (-1, 0), (0, 1), (0, -1),
    ]

    # Gradient vectors for 3D
    _GRAD3 = [
        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    ]

    def _initialize(self) -> None:
        """Generate permutation tables."""
        # Initialize permutation
        perm = list(range(256))

        # Shuffle with seed
        state = self._seed
        for i in range(255, 0, -1):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            j = state % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]

        # Double and create mod12 version
        self._perm = perm + perm
        self._perm_mod12 = [p % 12 for p in self._perm]
        self._initialized = True

    def _dot2(self, grad: Tuple[int, int], x: float, y: float) -> float:
        """2D dot product."""
        return grad[0] * x + grad[1] * y

    def _dot3(self, grad: Tuple[int, int, int], x: float, y: float, z: float) -> float:
        """3D dot product."""
        return grad[0] * x + grad[1] * y + grad[2] * z

    def sample(self, x: float, y: float) -> float:
        """
        Sample 2D simplex noise.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Apply frequency and offset
        x = x * self._settings.frequency + self._settings.offset[0]
        y = y * self._settings.frequency + self._settings.offset[1]

        # Skew input to get simplex cell
        s = (x + y) * self._F2
        i = int(math.floor(x + s))
        j = int(math.floor(y + s))

        # Unskew to get cell origin
        t = (i + j) * self._G2
        x0 = x - (i - t)
        y0 = y - (j - t)

        # Determine which simplex we're in
        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1

        # Offsets for middle and last corners
        x1 = x0 - i1 + self._G2
        y1 = y0 - j1 + self._G2
        x2 = x0 - 1.0 + 2.0 * self._G2
        y2 = y0 - 1.0 + 2.0 * self._G2

        # Hash coordinates
        ii = i & 255
        jj = j & 255

        # Calculate contributions from three corners
        n0 = n1 = n2 = 0.0

        t0 = 0.5 - x0 * x0 - y0 * y0
        if t0 >= 0:
            gi0 = self._perm[ii + self._perm[jj]] & 7
            t0 *= t0
            n0 = t0 * t0 * self._dot2(self._GRAD2[gi0], x0, y0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        if t1 >= 0:
            gi1 = self._perm[ii + i1 + self._perm[jj + j1]] & 7
            t1 *= t1
            n1 = t1 * t1 * self._dot2(self._GRAD2[gi1], x1, y1)

        t2 = 0.5 - x2 * x2 - y2 * y2
        if t2 >= 0:
            gi2 = self._perm[ii + 1 + self._perm[jj + 1]] & 7
            t2 *= t2
            n2 = t2 * t2 * self._dot2(self._GRAD2[gi2], x2, y2)

        # Scale to [-1, 1]
        value = 70.0 * (n0 + n1 + n2)

        return value * self._settings.amplitude

    def sample_3d(self, x: float, y: float, z: float) -> float:
        """
        Sample 3D simplex noise.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Apply frequency
        x = x * self._settings.frequency + self._settings.offset[0]
        y = y * self._settings.frequency + self._settings.offset[1]
        z = z * self._settings.frequency

        # Skew to simplex cell
        s = (x + y + z) * self._F3
        i = int(math.floor(x + s))
        j = int(math.floor(y + s))
        k = int(math.floor(z + s))

        # Unskew
        t = (i + j + k) * self._G3
        x0 = x - (i - t)
        y0 = y - (j - t)
        z0 = z - (k - t)

        # Determine simplex
        if x0 >= y0:
            if y0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 1, 0
            elif x0 >= z0:
                i1, j1, k1, i2, j2, k2 = 1, 0, 0, 1, 0, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 1, 0, 1
        else:
            if y0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 0, 1, 0, 1, 1
            elif x0 < z0:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 0, 1, 1
            else:
                i1, j1, k1, i2, j2, k2 = 0, 1, 0, 1, 1, 0

        # Offsets
        x1 = x0 - i1 + self._G3
        y1 = y0 - j1 + self._G3
        z1 = z0 - k1 + self._G3
        x2 = x0 - i2 + 2.0 * self._G3
        y2 = y0 - j2 + 2.0 * self._G3
        z2 = z0 - k2 + 2.0 * self._G3
        x3 = x0 - 1.0 + 3.0 * self._G3
        y3 = y0 - 1.0 + 3.0 * self._G3
        z3 = z0 - 1.0 + 3.0 * self._G3

        # Hash
        ii = i & 255
        jj = j & 255
        kk = k & 255

        # Contributions
        n0 = n1 = n2 = n3 = 0.0

        t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
        if t0 >= 0:
            gi0 = self._perm_mod12[ii + self._perm[jj + self._perm[kk]]]
            t0 *= t0
            n0 = t0 * t0 * self._dot3(self._GRAD3[gi0], x0, y0, z0)

        t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
        if t1 >= 0:
            gi1 = self._perm_mod12[ii + i1 + self._perm[jj + j1 + self._perm[kk + k1]]]
            t1 *= t1
            n1 = t1 * t1 * self._dot3(self._GRAD3[gi1], x1, y1, z1)

        t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
        if t2 >= 0:
            gi2 = self._perm_mod12[ii + i2 + self._perm[jj + j2 + self._perm[kk + k2]]]
            t2 *= t2
            n2 = t2 * t2 * self._dot3(self._GRAD3[gi2], x2, y2, z2)

        t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
        if t3 >= 0:
            gi3 = self._perm_mod12[ii + 1 + self._perm[jj + 1 + self._perm[kk + 1]]]
            t3 *= t3
            n3 = t3 * t3 * self._dot3(self._GRAD3[gi3], x3, y3, z3)

        # Scale to [-1, 1]
        value = 32.0 * (n0 + n1 + n2 + n3)

        return value * self._settings.amplitude


class WorleyNoise(NoiseGenerator):
    """
    Worley (cellular/Voronoi) noise generator.

    Creates cell-based noise useful for organic textures.
    Supports multiple distance metrics and return types.
    Output range: [0, 1] (normalized based on return type)
    """

    def __init__(
        self,
        seed: int = 0,
        settings: Optional[NoiseSettings] = None,
        distance_type: str = "euclidean",
        return_type: str = "f1",
    ) -> None:
        """
        Initialize Worley noise generator.

        Args:
            seed: Random seed
            settings: Noise settings
            distance_type: "euclidean", "manhattan", or "chebyshev"
            return_type: "f1" (nearest), "f2" (second nearest), or "f2-f1" (edge)
        """
        self._distance_type = distance_type
        self._return_type = return_type
        self._validate_params()
        super().__init__(seed, settings)

    def _validate_params(self) -> None:
        """Validate distance and return type parameters."""
        valid_distances = {"euclidean", "manhattan", "chebyshev"}
        if self._distance_type not in valid_distances:
            raise ValueError(
                f"Invalid distance_type '{self._distance_type}'. "
                f"Must be one of {valid_distances}"
            )

        valid_returns = {"f1", "f2", "f2-f1"}
        if self._return_type not in valid_returns:
            raise ValueError(
                f"Invalid return_type '{self._return_type}'. "
                f"Must be one of {valid_returns}"
            )

    @property
    def distance_type(self) -> str:
        """Get the distance metric type."""
        return self._distance_type

    @property
    def return_type(self) -> str:
        """Get the return value type."""
        return self._return_type

    def _initialize(self) -> None:
        """Initialize (no special state needed)."""
        self._initialized = True

    def _hash_cell(self, ix: int, iy: int) -> int:
        """Hash cell coordinates to get consistent random value."""
        return self._hash(ix, iy)

    def _get_feature_point(self, ix: int, iy: int) -> Tuple[float, float]:
        """Get the feature point within a cell."""
        h = self._hash_cell(ix, iy)
        # Use hash to generate point within cell [0, 1]
        px = ((h >> 0) & 0xFFFF) / 65535.0
        py = ((h >> 16) & 0xFFFF) / 65535.0
        return (ix + px, iy + py)

    def _distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance based on configured metric."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if self._distance_type == "euclidean":
            return math.sqrt(dx * dx + dy * dy)
        elif self._distance_type == "manhattan":
            return dx + dy
        else:  # chebyshev
            return max(dx, dy)

    def sample(self, x: float, y: float) -> float:
        """
        Sample 2D Worley noise.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Apply frequency and offset
        x = x * self._settings.frequency + self._settings.offset[0]
        y = y * self._settings.frequency + self._settings.offset[1]

        # Current cell
        ix = int(math.floor(x))
        iy = int(math.floor(y))

        # Find two closest distances
        f1 = float("inf")
        f2 = float("inf")

        # Search 3x3 neighborhood
        for di in range(-1, 2):
            for dj in range(-1, 2):
                cx, cy = ix + di, iy + dj
                px, py = self._get_feature_point(cx, cy)
                d = self._distance(x, y, px, py)

                if d < f1:
                    f2 = f1
                    f1 = d
                elif d < f2:
                    f2 = d

        # Determine return value
        if self._return_type == "f1":
            value = f1
        elif self._return_type == "f2":
            value = f2
        else:  # f2-f1
            value = f2 - f1

        # Normalize to [-1, 1] (approximate based on typical range)
        # f1 typically ranges 0 to ~0.7, f2-f1 ranges 0 to ~0.5
        if self._return_type == "f2-f1":
            value = value * 4.0 - 1.0
        else:
            value = value * 2.0 - 1.0

        # Clamp and apply amplitude
        value = max(-1.0, min(1.0, value))
        return value * self._settings.amplitude


class ValueNoise(NoiseGenerator):
    """
    Value noise generator.

    Random values at grid points with smooth interpolation.
    Simpler than Perlin but can have more visible grid artifacts.
    Output range: [-1, 1]
    """

    def __init__(self, seed: int = 0, settings: Optional[NoiseSettings] = None) -> None:
        self._value_table: List[float] = []
        super().__init__(seed, settings)

    def _initialize(self) -> None:
        """Generate value lookup table."""
        table_size = 256
        self._value_table = []

        state = self._seed
        for _ in range(table_size):
            # LCG to generate values in [-1, 1]
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            value = (state / 0x7FFFFFFF) * 2.0 - 1.0
            self._value_table.append(value)

        # Double for wrapping
        self._value_table = self._value_table + self._value_table
        self._initialized = True

    def _get_value(self, ix: int, iy: int) -> float:
        """Get value at integer grid point."""
        # Use permutation-style lookup with integer indexing
        iy_idx = iy & 255
        ix_idx = (ix + int(iy_idx * 7919)) & 255  # Use prime multiplier for mixing
        return self._value_table[ix_idx]

    @staticmethod
    def _smoothstep(t: float) -> float:
        """Smoothstep interpolation factor."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def sample(self, x: float, y: float) -> float:
        """
        Sample 2D value noise.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Apply frequency and offset
        x = x * self._settings.frequency + self._settings.offset[0]
        y = y * self._settings.frequency + self._settings.offset[1]

        # Grid coordinates
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1

        # Fractional part with smoothstep
        sx = self._smoothstep(x - x0)
        sy = self._smoothstep(y - y0)

        # Get values at corners
        v00 = self._get_value(x0, y0)
        v10 = self._get_value(x1, y0)
        v01 = self._get_value(x0, y1)
        v11 = self._get_value(x1, y1)

        # Bilinear interpolation
        v0 = v00 + sx * (v10 - v00)
        v1 = v01 + sx * (v11 - v01)
        value = v0 + sy * (v1 - v0)

        return value * self._settings.amplitude


class WhiteNoise(NoiseGenerator):
    """
    White noise generator.

    Pure random noise with no coherence.
    Useful as a base for other effects or for testing.
    Output range: [-1, 1]
    """

    def _initialize(self) -> None:
        """Initialize (no special state needed)."""
        self._initialized = True

    def sample(self, x: float, y: float) -> float:
        """
        Sample 2D white noise.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range [-1, 1]
        """
        # Hash coordinates to get deterministic random value
        # Convert to integers with high precision
        ix = int(x * 1000000) & 0xFFFFFF
        iy = int(y * 1000000) & 0xFFFFFF

        h = self._hash(ix, iy)
        value = (h / 0x7FFFFFFF) * 2.0 - 1.0

        return value * self._settings.amplitude


class FractalNoise:
    """
    Fractal (layered octave) noise generator.

    Combines multiple octaves of a base noise for more natural results.
    Also known as fBm (fractal Brownian motion).
    """

    def __init__(
        self,
        base_noise: NoiseGenerator,
        octaves: int = 4,
        lacunarity: float = 2.0,
        persistence: float = 0.5,
    ) -> None:
        """
        Initialize fractal noise.

        Args:
            base_noise: The underlying noise generator
            octaves: Number of noise layers
            lacunarity: Frequency multiplier per octave
            persistence: Amplitude multiplier per octave
        """
        self._base_noise = base_noise
        self._octaves = octaves
        self._lacunarity = lacunarity
        self._persistence = persistence

        if octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {octaves}")
        if lacunarity <= 0:
            raise ValueError(f"lacunarity must be > 0, got {lacunarity}")
        if not 0 < persistence <= 1:
            raise ValueError(f"persistence must be in (0, 1], got {persistence}")

    @property
    def base_noise(self) -> NoiseGenerator:
        """Get the base noise generator."""
        return self._base_noise

    @property
    def octaves(self) -> int:
        """Get the number of octaves."""
        return self._octaves

    @property
    def lacunarity(self) -> float:
        """Get the lacunarity (frequency multiplier)."""
        return self._lacunarity

    @property
    def persistence(self) -> float:
        """Get the persistence (amplitude multiplier)."""
        return self._persistence

    def sample(self, x: float, y: float) -> float:
        """
        Sample 2D fractal noise.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Noise value in range [-1, 1] (approximate)
        """
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_amplitude = 0.0

        for _ in range(self._octaves):
            total += self._base_noise.sample(x * frequency, y * frequency) * amplitude
            max_amplitude += amplitude
            amplitude *= self._persistence
            frequency *= self._lacunarity

        # Normalize
        return total / max_amplitude

    def sample_3d(self, x: float, y: float, z: float) -> float:
        """
        Sample 3D fractal noise.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate

        Returns:
            Noise value in range [-1, 1] (approximate)
        """
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_amplitude = 0.0

        for _ in range(self._octaves):
            total += self._base_noise.sample_3d(
                x * frequency, y * frequency, z * frequency
            ) * amplitude
            max_amplitude += amplitude
            amplitude *= self._persistence
            frequency *= self._lacunarity

        return total / max_amplitude


class NoiseMap:
    """
    2D array of noise values for terrain and texture generation.

    Provides utilities for generation, sampling, and manipulation.
    """

    def __init__(
        self,
        width: int,
        height: int,
        settings: Optional[NoiseSettings] = None,
    ) -> None:
        """
        Initialize a noise map.

        Args:
            width: Map width in samples
            height: Map height in samples
            settings: Noise generation settings
        """
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if height < 1:
            raise ValueError(f"height must be >= 1, got {height}")

        self._width = width
        self._height = height
        self._settings = settings or NoiseSettings()
        self._data: List[List[float]] = [[0.0] * width for _ in range(height)]

    @property
    def width(self) -> int:
        """Get the map width."""
        return self._width

    @property
    def height(self) -> int:
        """Get the map height."""
        return self._height

    @property
    def settings(self) -> NoiseSettings:
        """Get the noise settings."""
        return self._settings

    def generate(self, generator: Optional[NoiseGenerator] = None) -> None:
        """
        Generate noise values for the entire map.

        Args:
            generator: Noise generator to use (creates default if None)
        """
        if generator is None:
            generator = self._create_generator()

        for y in range(self._height):
            for x in range(self._width):
                # Normalize coordinates to [0, 1]
                nx = x / self._width
                ny = y / self._height
                self._data[y][x] = generator.sample(nx, ny)

    def _create_generator(self) -> NoiseGenerator:
        """Create a noise generator based on settings."""
        generators = {
            NoiseType.PERLIN: PerlinNoise,
            NoiseType.SIMPLEX: SimplexNoise,
            NoiseType.WORLEY: WorleyNoise,
            NoiseType.VALUE: ValueNoise,
            NoiseType.WHITE: WhiteNoise,
        }

        gen_class = generators.get(self._settings.noise_type, PerlinNoise)
        return gen_class(self._settings.seed, self._settings)

    def get_value(self, x: float, y: float) -> float:
        """
        Get interpolated value at floating-point coordinates.

        Args:
            x: X coordinate in [0, width)
            y: Y coordinate in [0, height)

        Returns:
            Interpolated noise value
        """
        # Clamp coordinates
        x = max(0.0, min(x, self._width - 1.001))
        y = max(0.0, min(y, self._height - 1.001))

        # Integer and fractional parts
        x0 = int(x)
        y0 = int(y)
        x1 = min(x0 + 1, self._width - 1)
        y1 = min(y0 + 1, self._height - 1)

        fx = x - x0
        fy = y - y0

        # Bilinear interpolation
        v00 = self._data[y0][x0]
        v10 = self._data[y0][x1]
        v01 = self._data[y1][x0]
        v11 = self._data[y1][x1]

        v0 = v00 + fx * (v10 - v00)
        v1 = v01 + fx * (v11 - v01)

        return v0 + fy * (v1 - v0)

    def get_raw(self, x: int, y: int) -> float:
        """
        Get raw value at integer coordinates.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Raw noise value
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds")
        return self._data[y][x]

    def set_raw(self, x: int, y: int, value: float) -> None:
        """
        Set raw value at integer coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            value: Value to set
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError(f"Coordinates ({x}, {y}) out of bounds")
        self._data[y][x] = value

    def normalize(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        """
        Normalize all values to a specified range.

        Args:
            min_val: Target minimum value
            max_val: Target maximum value
        """
        # Find current range
        current_min = float("inf")
        current_max = float("-inf")

        for row in self._data:
            for value in row:
                current_min = min(current_min, value)
                current_max = max(current_max, value)

        # Avoid division by zero
        current_range = current_max - current_min
        if current_range < 1e-10:
            return

        target_range = max_val - min_val

        # Normalize
        for y in range(self._height):
            for x in range(self._width):
                normalized = (self._data[y][x] - current_min) / current_range
                self._data[y][x] = min_val + normalized * target_range

    def apply_curve(self, curve_func: Callable[[float], float]) -> None:
        """
        Apply a curve function to all values.

        Args:
            curve_func: Function that transforms values
        """
        for y in range(self._height):
            for x in range(self._width):
                self._data[y][x] = curve_func(self._data[y][x])

    def to_list(self) -> List[List[float]]:
        """Get a copy of the internal data as a 2D list."""
        return [row.copy() for row in self._data]

    def from_list(self, data: List[List[float]]) -> None:
        """
        Set internal data from a 2D list.

        Args:
            data: 2D list of values (must match dimensions)
        """
        if len(data) != self._height:
            raise ValueError(f"Data height {len(data)} != map height {self._height}")
        for i, row in enumerate(data):
            if len(row) != self._width:
                raise ValueError(
                    f"Row {i} width {len(row)} != map width {self._width}"
                )

        self._data = [row.copy() for row in data]


# Factory function for creating noise generators
def create_noise_generator(
    noise_type: NoiseType,
    seed: int = 0,
    settings: Optional[NoiseSettings] = None,
    **kwargs,
) -> NoiseGenerator:
    """
    Factory function to create noise generators.

    Args:
        noise_type: Type of noise to generate
        seed: Random seed
        settings: Noise settings
        **kwargs: Additional arguments for specific noise types

    Returns:
        A configured noise generator
    """
    if settings is None:
        settings = NoiseSettings(noise_type=noise_type, seed=seed)

    generators = {
        NoiseType.PERLIN: PerlinNoise,
        NoiseType.SIMPLEX: SimplexNoise,
        NoiseType.VALUE: ValueNoise,
        NoiseType.WHITE: WhiteNoise,
    }

    if noise_type == NoiseType.WORLEY:
        return WorleyNoise(
            seed=seed,
            settings=settings,
            distance_type=kwargs.get("distance_type", "euclidean"),
            return_type=kwargs.get("return_type", "f1"),
        )

    gen_class = generators.get(noise_type, PerlinNoise)
    return gen_class(seed, settings)


__all__ = [
    "NoiseType",
    "NoiseSettings",
    "NoiseGenerator",
    "PerlinNoise",
    "SimplexNoise",
    "WorleyNoise",
    "ValueNoise",
    "WhiteNoise",
    "FractalNoise",
    "NoiseMap",
    "create_noise_generator",
]
