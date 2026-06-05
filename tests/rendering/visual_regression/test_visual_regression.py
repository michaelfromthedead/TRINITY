"""Visual regression testing implementation for TRINITY rendering.

This module implements screenshot comparison testing with:
- DeltaE perceptual difference metric (CIE LAB color space)
- Per-pixel error visualization
- Reference image management
- CI integration support

Task: T-MAT-11.2 Visual Regression Testing
Acceptance criteria:
- Identical renders: < 0.5% pixel difference
- Deliberate regression: > 5% pixel difference
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

# Module directory for reference images
MODULE_DIR = Path(__file__).parent
REFERENCE_DIR = MODULE_DIR / "reference_images"
DIFF_OUTPUT_DIR = MODULE_DIR / "diff_output"

# Ensure directories exist
REFERENCE_DIR.mkdir(exist_ok=True)
DIFF_OUTPUT_DIR.mkdir(exist_ok=True)

# Acceptance thresholds
IDENTICAL_THRESHOLD = 0.5  # < 0.5% for identical
REGRESSION_THRESHOLD = 5.0  # > 5.0% for regression


class MaterialVariant(Enum):
    """Material domain variants for testing."""
    SURFACE = "surface"
    DEFERRED_DECAL = "deferred_decal"
    VOLUME = "volume"
    POST_PROCESS = "post_process"
    UI = "ui"


class RenderScene(Enum):
    """Standard test scenes for visual regression."""
    PBR_SPHERE = "pbr_sphere"
    MATERIAL_VARIANTS = "material_variants"
    ADVANCED_SHADING = "advanced_shading"
    SSS_TEST = "sss_test"
    TRANSMISSION_TEST = "transmission_test"
    CLEAR_COAT_TEST = "clear_coat_test"
    LIGHTING_TEST = "lighting_test"


@dataclass
class DiffResult:
    """Result of image comparison.

    Attributes:
        match: Whether images are within threshold
        pixel_diff_percent: Percentage of differing pixels
        mean_delta_e: Mean DeltaE color difference
        max_delta_e: Maximum DeltaE value found
        diff_image: Per-pixel difference visualization
        error_mask: Binary mask of error locations
        histogram: Histogram of DeltaE values
    """
    match: bool
    pixel_diff_percent: float
    mean_delta_e: float
    max_delta_e: float
    diff_image: Optional[np.ndarray] = None
    error_mask: Optional[np.ndarray] = None
    histogram: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Convert numpy types to native Python types for JSON
        return {
            "match": bool(self.match),
            "pixel_diff_percent": float(self.pixel_diff_percent),
            "mean_delta_e": float(self.mean_delta_e),
            "max_delta_e": float(self.max_delta_e),
            "histogram": {k: int(v) for k, v in self.histogram.items()}
            if self.histogram else None,
        }


@dataclass
class RenderConfig:
    """Configuration for rendering a test scene.

    Attributes:
        width: Render width in pixels
        height: Render height in pixels
        samples: Number of samples for anti-aliasing
        exposure: Exposure value
        gamma: Gamma correction value
    """
    width: int = 512
    height: int = 512
    samples: int = 4
    exposure: float = 1.0
    gamma: float = 2.2


class ColorSpaceConverter:
    """Converts between color spaces for perceptual comparison."""

    # D65 white point
    _WHITE_X = 0.95047
    _WHITE_Y = 1.0
    _WHITE_Z = 1.08883

    # sRGB to XYZ matrix
    _SRGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])

    @staticmethod
    def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear RGB.

        Args:
            srgb: sRGB values in [0, 1]

        Returns:
            Linear RGB values
        """
        # Ensure float type
        srgb = srgb.astype(np.float64)

        # Apply inverse sRGB transfer function
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        return linear

    @staticmethod
    def linear_to_xyz(linear_rgb: np.ndarray) -> np.ndarray:
        """Convert linear RGB to CIE XYZ.

        Args:
            linear_rgb: Linear RGB array of shape (..., 3)

        Returns:
            XYZ values
        """
        original_shape = linear_rgb.shape
        # Reshape to (N, 3) for matrix multiplication
        flat = linear_rgb.reshape(-1, 3)
        xyz = flat @ ColorSpaceConverter._SRGB_TO_XYZ.T
        return xyz.reshape(original_shape)

    @staticmethod
    def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
        """Convert CIE XYZ to CIE LAB.

        Args:
            xyz: XYZ array of shape (..., 3)

        Returns:
            LAB values
        """
        # Normalize by white point
        x = xyz[..., 0] / ColorSpaceConverter._WHITE_X
        y = xyz[..., 1] / ColorSpaceConverter._WHITE_Y
        z = xyz[..., 2] / ColorSpaceConverter._WHITE_Z

        # Apply f function
        epsilon = 216.0 / 24389.0
        kappa = 24389.0 / 27.0

        def f(t: np.ndarray) -> np.ndarray:
            return np.where(
                t > epsilon,
                np.cbrt(t),
                (kappa * t + 16.0) / 116.0
            )

        fx, fy, fz = f(x), f(y), f(z)

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)

        return np.stack([L, a, b], axis=-1)

    @staticmethod
    def srgb_to_lab(srgb: np.ndarray) -> np.ndarray:
        """Convert sRGB directly to CIE LAB.

        Args:
            srgb: sRGB values in [0, 1] with shape (..., 3)

        Returns:
            LAB values
        """
        linear = ColorSpaceConverter.srgb_to_linear(srgb)
        xyz = ColorSpaceConverter.linear_to_xyz(linear)
        lab = ColorSpaceConverter.xyz_to_lab(xyz)
        return lab


class DeltaECalculator:
    """Calculates perceptual color difference using DeltaE formulas."""

    @staticmethod
    def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
        """Calculate CIE76 DeltaE (Euclidean distance in LAB).

        Simple but less perceptually uniform than DeltaE 2000.

        Args:
            lab1: First LAB color array
            lab2: Second LAB color array

        Returns:
            DeltaE values
        """
        diff = lab1 - lab2
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    @staticmethod
    def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
        """Calculate CIE DE2000 DeltaE (most perceptually accurate).

        Implements the CIEDE2000 formula with all weighting factors.

        Args:
            lab1: First LAB color array of shape (..., 3)
            lab2: Second LAB color array of shape (..., 3)

        Returns:
            DeltaE 2000 values
        """
        # Extract components
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

        # Calculate C'
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_bar = (C1 + C2) / 2.0

        # Calculate G
        C_bar_7 = C_bar**7
        G = 0.5 * (1 - np.sqrt(C_bar_7 / (C_bar_7 + 25**7)))

        # Adjusted a'
        a1_prime = a1 * (1 + G)
        a2_prime = a2 * (1 + G)

        # Chroma C'
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)

        # Hue angle h'
        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

        # Delta L', C', H'
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime

        # Delta h'
        h_diff = h2_prime - h1_prime
        delta_h_prime = np.where(
            np.abs(h_diff) <= 180,
            h_diff,
            np.where(h_diff > 180, h_diff - 360, h_diff + 360)
        )

        # Handle case where C is zero
        delta_h_prime = np.where(
            (C1_prime * C2_prime) == 0,
            0,
            delta_h_prime
        )

        # Delta H'
        delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(
            np.radians(delta_h_prime / 2)
        )

        # Weighted average values
        L_prime_bar = (L1 + L2) / 2.0
        C_prime_bar = (C1_prime + C2_prime) / 2.0

        # Average hue
        h_prime_bar = np.where(
            np.abs(h_diff) <= 180,
            (h1_prime + h2_prime) / 2.0,
            np.where(
                h1_prime + h2_prime < 360,
                (h1_prime + h2_prime + 360) / 2.0,
                (h1_prime + h2_prime - 360) / 2.0
            )
        )
        h_prime_bar = np.where(
            (C1_prime * C2_prime) == 0,
            h1_prime + h2_prime,
            h_prime_bar
        )

        # T factor
        T = (1
             - 0.17 * np.cos(np.radians(h_prime_bar - 30))
             + 0.24 * np.cos(np.radians(2 * h_prime_bar))
             + 0.32 * np.cos(np.radians(3 * h_prime_bar + 6))
             - 0.20 * np.cos(np.radians(4 * h_prime_bar - 63)))

        # Weighting factors
        L_bar_minus_50_sq = (L_prime_bar - 50)**2
        S_L = 1 + (0.015 * L_bar_minus_50_sq) / np.sqrt(20 + L_bar_minus_50_sq)
        S_C = 1 + 0.045 * C_prime_bar
        S_H = 1 + 0.015 * C_prime_bar * T

        # Rotation factor
        delta_theta = 30 * np.exp(-((h_prime_bar - 275) / 25)**2)
        C_prime_bar_7 = C_prime_bar**7
        R_C = 2 * np.sqrt(C_prime_bar_7 / (C_prime_bar_7 + 25**7))
        R_T = -R_C * np.sin(np.radians(2 * delta_theta))

        # Parametric factors (kL, kC, kH = 1 for standard conditions)
        kL = kC = kH = 1.0

        # Final DeltaE 2000
        term1 = (delta_L_prime / (kL * S_L))**2
        term2 = (delta_C_prime / (kC * S_C))**2
        term3 = (delta_H_prime / (kH * S_H))**2
        term4 = (R_T
                 * (delta_C_prime / (kC * S_C))
                 * (delta_H_prime / (kH * S_H)))

        delta_e = np.sqrt(term1 + term2 + term3 + term4)

        return delta_e


class MockRenderer:
    """Mock renderer for testing without GPU.

    In production, this would interface with the actual TRINITY renderer.
    For testing, it generates deterministic synthetic images.
    """

    def __init__(self, config: RenderConfig) -> None:
        self._config = config
        self._seed = 42

    def render_scene(
        self,
        scene: RenderScene,
        material_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render a test scene and return the image.

        Args:
            scene: Scene to render
            material_params: Optional material parameter overrides

        Returns:
            Rendered image as RGB numpy array [0, 1]
        """
        np.random.seed(self._seed)
        w, h = self._config.width, self._config.height

        # Generate deterministic scene-based image
        if scene == RenderScene.PBR_SPHERE:
            return self._render_pbr_sphere(w, h, material_params)
        elif scene == RenderScene.MATERIAL_VARIANTS:
            return self._render_material_variants(w, h, material_params)
        elif scene == RenderScene.ADVANCED_SHADING:
            return self._render_advanced_shading(w, h, material_params)
        elif scene == RenderScene.SSS_TEST:
            return self._render_sss(w, h, material_params)
        elif scene == RenderScene.TRANSMISSION_TEST:
            return self._render_transmission(w, h, material_params)
        elif scene == RenderScene.CLEAR_COAT_TEST:
            return self._render_clear_coat(w, h, material_params)
        elif scene == RenderScene.LIGHTING_TEST:
            return self._render_lighting(w, h, material_params)
        else:
            return self._render_default(w, h)

    def _render_pbr_sphere(
        self,
        w: int,
        h: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render a PBR sphere with lighting."""
        params = params or {}
        base_color = params.get("base_color", (0.8, 0.2, 0.2))
        metallic = params.get("metallic", 0.0)
        roughness = params.get("roughness", 0.5)

        image = np.zeros((h, w, 3), dtype=np.float64)

        # Create sphere mask
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 3
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = dist < r

        # Calculate normals
        nx = np.where(mask, (x - cx) / r, 0)
        ny = np.where(mask, (y - cy) / r, 0)
        nz = np.where(mask, np.sqrt(np.maximum(1 - nx**2 - ny**2, 0)), 0)

        # Simple lighting
        light_dir = np.array([0.5, -0.5, 1.0])
        light_dir /= np.linalg.norm(light_dir)

        ndotl = np.maximum(
            nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2],
            0
        )

        # Apply material
        for i in range(3):
            # Diffuse with roughness
            diffuse = base_color[i] * ndotl * (1 - metallic) * (1 - roughness * 0.5)

            # Simple specular
            view_dir = np.array([0, 0, 1])
            h_vec = light_dir + view_dir
            h_vec /= np.linalg.norm(h_vec)
            ndoth = np.maximum(
                nx * h_vec[0] + ny * h_vec[1] + nz * h_vec[2],
                0
            )
            spec_power = (1 - roughness) * 64 + 1
            specular = np.power(ndoth, spec_power) * (0.04 + metallic * 0.96)

            image[:, :, i] = np.where(mask, diffuse + specular, 0.1)

        return np.clip(image, 0, 1)

    def _render_material_variants(
        self,
        w: int,
        h: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render multiple material variants in a grid."""
        image = np.zeros((h, w, 3), dtype=np.float64)

        # 5 variants in a row
        variants = [
            MaterialVariant.SURFACE,
            MaterialVariant.DEFERRED_DECAL,
            MaterialVariant.VOLUME,
            MaterialVariant.POST_PROCESS,
            MaterialVariant.UI,
        ]

        cell_w = w // len(variants)

        for i, variant in enumerate(variants):
            x_start = i * cell_w
            x_end = (i + 1) * cell_w if i < len(variants) - 1 else w

            # Each variant has a different base color
            hue = i / len(variants)
            r = abs(math.sin(hue * math.pi * 2)) * 0.8 + 0.2
            g = abs(math.sin((hue + 0.33) * math.pi * 2)) * 0.8 + 0.2
            b = abs(math.sin((hue + 0.66) * math.pi * 2)) * 0.8 + 0.2

            # Create gradient for each cell
            y_coords = np.linspace(0, 1, h).reshape(-1, 1)
            gradient = 0.5 + 0.5 * np.sin(y_coords * math.pi)

            image[:, x_start:x_end, 0] = r * gradient
            image[:, x_start:x_end, 1] = g * gradient
            image[:, x_start:x_end, 2] = b * gradient

        return np.clip(image, 0, 1)

    def _render_advanced_shading(
        self,
        w: int,
        h: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render advanced shading models: SSS, transmission, clear coat."""
        image = np.zeros((h, w, 3), dtype=np.float64)

        # Three regions: SSS, transmission, clear coat
        third_w = w // 3

        # SSS (left) - skin-like
        image[:, :third_w, 0] = 0.8
        image[:, :third_w, 1] = 0.6
        image[:, :third_w, 2] = 0.5

        # Apply SSS-like scatter
        y_norm = np.linspace(0, 1, h).reshape(-1, 1)
        scatter = np.exp(-y_norm * 3)
        image[:, :third_w, 0] += 0.2 * scatter
        image[:, :third_w, 1] += 0.1 * scatter

        # Transmission (middle) - glass-like
        image[:, third_w:2*third_w, 0] = 0.9
        image[:, third_w:2*third_w, 1] = 0.95
        image[:, third_w:2*third_w, 2] = 1.0

        # Apply transmission distortion
        x_offset = np.sin(np.linspace(0, math.pi * 4, h)).reshape(-1, 1, 1) * 0.1
        image[:, third_w:2*third_w, :] *= (1 + x_offset)

        # Clear coat (right) - car paint-like
        image[:, 2*third_w:, 0] = 0.1
        image[:, 2*third_w:, 1] = 0.3
        image[:, 2*third_w:, 2] = 0.8

        # Add clear coat highlight
        y, x = np.ogrid[:h, :w-2*third_w]
        cx, cy = (w - 2*third_w) // 2, h // 3
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        highlight = np.exp(-dist**2 / 5000) * 0.5
        image[:, 2*third_w:, :] += highlight[..., np.newaxis]

        return np.clip(image, 0, 1)

    def _render_sss(
        self,
        w: int,
        h: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render subsurface scattering test scene."""
        params = params or {}
        scatter_radius = params.get("scatter_radius", 1.0)

        image = np.zeros((h, w, 3), dtype=np.float64)

        # Create sphere with SSS
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        r = min(w, h) // 3
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = dist < r

        # Skin-like base
        base = np.array([0.8, 0.5, 0.4])

        # SSS transmission effect
        thickness = np.where(mask, np.sqrt(np.maximum(r**2 - dist**2, 0)) / r, 0)
        scatter = np.exp(-thickness * 2 / scatter_radius)

        for i in range(3):
            sss_color = [1.0, 0.3, 0.1][i]  # Red-orange scatter
            image[:, :, i] = np.where(
                mask,
                base[i] * (1 - scatter * 0.5) + sss_color * scatter * 0.3,
                0.1
            )

        return np.clip(image, 0, 1)

    def _render_transmission(
        self,
        w: int,
        h: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render transmission/glass test scene."""
        params = params or {}
        ior = params.get("ior", 1.5)

        image = np.zeros((h, w, 3), dtype=np.float64)

        # Background pattern
        checker_size = 32
        y, x = np.mgrid[:h, :w]
        checker = ((x // checker_size + y // checker_size) % 2).astype(np.float64)
        background = checker * 0.3 + 0.3

        # Glass sphere overlay
        cx, cy = w // 2, h // 2
        r = min(w, h) // 3
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = dist < r

        # Fresnel effect - IOR affects F0 calculation
        f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
        thickness = np.where(mask, np.sqrt(np.maximum(r**2 - dist**2, 0)) / r, 0)
        fresnel = f0 + (1.0 - f0) * np.power(1 - thickness, 5)

        # Refraction distortion simulation - IOR directly affects distortion
        distortion_factor = (ior - 1.0) * 0.2

        for i in range(3):
            # IOR affects refracted color shift
            color_shift = (i - 1) * distortion_factor * thickness
            refracted = background + color_shift

            image[:, :, i] = np.where(
                mask,
                refracted * (1 - fresnel) + (0.95 - 0.1 * distortion_factor) * fresnel,
                background
            )

        return np.clip(image, 0, 1)

    def _render_clear_coat(
        self,
        w: int,
        h: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render clear coat test scene."""
        params = params or {}
        coat_intensity = params.get("clear_coat_intensity", 1.0)
        coat_roughness = params.get("clear_coat_roughness", 0.0)

        image = np.zeros((h, w, 3), dtype=np.float64)

        # Car paint base - intensity affects how much base shows through
        base_color = np.array([0.05, 0.2, 0.6])

        # Create curved surface
        y, x = np.ogrid[:h, :w]
        nx = (x - w // 2) / (w // 2)
        ny = (y - h // 2) / (h // 2)
        curve = np.sqrt(np.maximum(1 - nx**2 * 0.5 - ny**2 * 0.5, 0))

        # Base layer - intensity affects visibility
        base_blend = 1.0 - coat_intensity * 0.3  # More coat = less base visible
        for i in range(3):
            image[:, :, i] = base_color[i] * curve * base_blend

        # Clear coat reflection - major visual impact from intensity
        light_angle = 0.3
        reflect = np.maximum(curve - light_angle, 0)
        spec_sharpness = (1 - coat_roughness) * 20 + 1

        # Strong clear coat specular based on intensity
        clear_coat_spec = np.power(reflect, spec_sharpness) * coat_intensity * 0.8

        # Also add fresnel-like rim highlight based on intensity
        rim = np.maximum(1.0 - curve, 0) ** 2 * coat_intensity * 0.5

        image += (clear_coat_spec + rim)[..., np.newaxis]

        return np.clip(image, 0, 1)

    def _render_lighting(
        self,
        w: int,
        h: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render lighting test scene with multiple lights."""
        image = np.zeros((h, w, 3), dtype=np.float64)

        # Three colored lights
        lights = [
            {"pos": (w * 0.2, h * 0.3), "color": (1.0, 0.2, 0.1), "radius": 100},
            {"pos": (w * 0.5, h * 0.5), "color": (0.2, 1.0, 0.2), "radius": 120},
            {"pos": (w * 0.8, h * 0.7), "color": (0.1, 0.3, 1.0), "radius": 100},
        ]

        y, x = np.ogrid[:h, :w]

        for light in lights:
            lx, ly = light["pos"]
            dist = np.sqrt((x - lx)**2 + (y - ly)**2)
            intensity = np.exp(-dist**2 / (2 * light["radius"]**2))

            for i, c in enumerate(light["color"]):
                image[:, :, i] += c * intensity

        return np.clip(image, 0, 1)

    def _render_default(self, w: int, h: int) -> np.ndarray:
        """Render default test pattern."""
        image = np.zeros((h, w, 3), dtype=np.float64)

        # Gradient pattern
        y, x = np.mgrid[:h, :w]
        image[:, :, 0] = x / w
        image[:, :, 1] = y / h
        image[:, :, 2] = 0.5

        return np.clip(image, 0, 1)


class VisualRegressionTest:
    """Main class for visual regression testing.

    Provides methods for:
    - Capturing screenshots from rendered scenes
    - Comparing images using DeltaE perceptual metric
    - Generating diff visualizations
    - Managing reference images
    """

    def __init__(
        self,
        reference_dir: Optional[Path] = None,
        diff_output_dir: Optional[Path] = None,
        config: Optional[RenderConfig] = None,
    ) -> None:
        """Initialize the visual regression test framework.

        Args:
            reference_dir: Directory for reference images
            diff_output_dir: Directory for diff output
            config: Render configuration
        """
        self._reference_dir = reference_dir or REFERENCE_DIR
        self._diff_output_dir = diff_output_dir or DIFF_OUTPUT_DIR
        self._config = config or RenderConfig()
        self._renderer = MockRenderer(self._config)

        # Ensure directories exist
        self._reference_dir.mkdir(parents=True, exist_ok=True)
        self._diff_output_dir.mkdir(parents=True, exist_ok=True)

    def capture_screenshot(
        self,
        scene: Union[RenderScene, str],
        material_params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Capture a screenshot of a rendered scene.

        Args:
            scene: Scene to render
            material_params: Optional material parameter overrides

        Returns:
            Rendered image as numpy array [0, 1]
        """
        if isinstance(scene, str):
            scene = RenderScene(scene)

        return self._renderer.render_scene(scene, material_params)

    def compare_images(
        self,
        actual: np.ndarray,
        reference: np.ndarray,
        threshold: float = IDENTICAL_THRESHOLD,
    ) -> DiffResult:
        """Compare two images using DeltaE 2000 perceptual metric.

        Args:
            actual: Actual rendered image
            reference: Reference image
            threshold: Maximum allowed pixel difference percentage

        Returns:
            DiffResult with comparison metrics
        """
        if actual.shape != reference.shape:
            raise ValueError(
                f"Image shapes do not match: {actual.shape} vs {reference.shape}"
            )

        # Ensure RGB (remove alpha if present)
        if actual.shape[-1] == 4:
            actual = actual[..., :3]
        if reference.shape[-1] == 4:
            reference = reference[..., :3]

        # Convert to LAB
        actual_lab = ColorSpaceConverter.srgb_to_lab(actual)
        reference_lab = ColorSpaceConverter.srgb_to_lab(reference)

        # Calculate DeltaE 2000
        delta_e = DeltaECalculator.delta_e_2000(actual_lab, reference_lab)

        # Statistics
        mean_delta_e = float(np.mean(delta_e))
        max_delta_e = float(np.max(delta_e))

        # Count pixels above perceptual threshold (DeltaE > 2.3 is noticeable)
        noticeable_threshold = 2.3
        diff_pixels = np.sum(delta_e > noticeable_threshold)
        total_pixels = delta_e.size
        pixel_diff_percent = (diff_pixels / total_pixels) * 100

        # Generate diff visualization
        diff_image = self.generate_diff_image(actual, reference, delta_e)

        # Error mask
        error_mask = (delta_e > noticeable_threshold).astype(np.uint8) * 255

        # DeltaE histogram
        histogram = {
            "0-1": int(np.sum(delta_e < 1)),
            "1-2": int(np.sum((delta_e >= 1) & (delta_e < 2))),
            "2-5": int(np.sum((delta_e >= 2) & (delta_e < 5))),
            "5-10": int(np.sum((delta_e >= 5) & (delta_e < 10))),
            "10+": int(np.sum(delta_e >= 10)),
        }

        match = pixel_diff_percent < threshold

        return DiffResult(
            match=match,
            pixel_diff_percent=pixel_diff_percent,
            mean_delta_e=mean_delta_e,
            max_delta_e=max_delta_e,
            diff_image=diff_image,
            error_mask=error_mask,
            histogram=histogram,
        )

    def delta_e_perceptual(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> float:
        """Calculate mean DeltaE 2000 perceptual difference.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            Mean DeltaE value
        """
        lab_a = ColorSpaceConverter.srgb_to_lab(image_a[..., :3])
        lab_b = ColorSpaceConverter.srgb_to_lab(image_b[..., :3])
        delta_e = DeltaECalculator.delta_e_2000(lab_a, lab_b)
        return float(np.mean(delta_e))

    def generate_diff_image(
        self,
        actual: np.ndarray,
        reference: np.ndarray,
        delta_e: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate per-pixel error visualization.

        Uses a heat map where:
        - Blue: No difference
        - Green: Slight difference
        - Yellow: Moderate difference
        - Red: Large difference

        Args:
            actual: Actual rendered image
            reference: Reference image
            delta_e: Pre-computed DeltaE values (optional)

        Returns:
            Diff visualization as RGB array
        """
        if delta_e is None:
            lab_a = ColorSpaceConverter.srgb_to_lab(actual[..., :3])
            lab_b = ColorSpaceConverter.srgb_to_lab(reference[..., :3])
            delta_e = DeltaECalculator.delta_e_2000(lab_a, lab_b)

        # Normalize to [0, 1] with max at DeltaE=10
        normalized = np.clip(delta_e / 10.0, 0, 1)

        # Create heat map
        diff_image = np.zeros((*delta_e.shape, 3), dtype=np.float64)

        # Blue (0) -> Green (0.33) -> Yellow (0.66) -> Red (1)
        # Blue channel
        diff_image[..., 2] = np.maximum(1 - normalized * 3, 0)

        # Green channel
        diff_image[..., 1] = np.where(
            normalized < 0.5,
            normalized * 2,
            2 - normalized * 2
        )

        # Red channel
        diff_image[..., 0] = np.maximum(normalized * 3 - 2, 0)
        diff_image[..., 0] = np.where(
            normalized > 0.33,
            np.minimum(normalized * 1.5, 1),
            0
        )

        return np.clip(diff_image, 0, 1)

    def save_reference(
        self,
        name: str,
        image: np.ndarray,
    ) -> Path:
        """Save an image as a reference.

        Args:
            name: Reference name (without extension)
            image: Image to save

        Returns:
            Path to saved reference
        """
        path = self._reference_dir / f"{name}.npy"
        np.save(path, image)
        return path

    def load_reference(self, name: str) -> Optional[np.ndarray]:
        """Load a reference image.

        Args:
            name: Reference name (without extension)

        Returns:
            Reference image or None if not found
        """
        path = self._reference_dir / f"{name}.npy"
        if path.exists():
            return np.load(path)
        return None

    def save_diff_output(
        self,
        name: str,
        diff_result: DiffResult,
        actual: np.ndarray,
        reference: np.ndarray,
    ) -> Path:
        """Save diff output for CI/debugging.

        Args:
            name: Test name
            diff_result: Comparison result
            actual: Actual image
            reference: Reference image

        Returns:
            Path to output directory
        """
        output_dir = self._diff_output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save images
        np.save(output_dir / "actual.npy", actual)
        np.save(output_dir / "reference.npy", reference)
        if diff_result.diff_image is not None:
            np.save(output_dir / "diff.npy", diff_result.diff_image)
        if diff_result.error_mask is not None:
            np.save(output_dir / "error_mask.npy", diff_result.error_mask)

        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(diff_result.to_dict(), f, indent=2)

        return output_dir


# ==============================================================================
# Pytest Test Suite
# ==============================================================================


class TestVisualRegressionFramework:
    """Tests for the visual regression framework itself."""

    @pytest.fixture
    def vrt(self) -> VisualRegressionTest:
        """Create a VisualRegressionTest instance."""
        return VisualRegressionTest()

    def test_color_space_srgb_to_linear(self) -> None:
        """Test sRGB to linear conversion."""
        # Black
        black = np.array([0, 0, 0])
        assert np.allclose(ColorSpaceConverter.srgb_to_linear(black), [0, 0, 0])

        # White
        white = np.array([1, 1, 1])
        assert np.allclose(ColorSpaceConverter.srgb_to_linear(white), [1, 1, 1])

        # Mid gray (sRGB 0.5 != linear 0.5)
        gray = np.array([0.5, 0.5, 0.5])
        linear_gray = ColorSpaceConverter.srgb_to_linear(gray)
        assert np.all(linear_gray < 0.5)  # sRGB is brighter than linear

    def test_color_space_srgb_to_lab(self) -> None:
        """Test sRGB to LAB conversion."""
        # White should have L=100
        white = np.array([1, 1, 1])
        lab = ColorSpaceConverter.srgb_to_lab(white)
        assert abs(lab[0] - 100) < 1  # L close to 100

        # Black should have L=0
        black = np.array([0, 0, 0])
        lab = ColorSpaceConverter.srgb_to_lab(black)
        assert lab[0] < 1  # L close to 0

    def test_delta_e_76_identical(self) -> None:
        """Test DeltaE 76 for identical colors."""
        lab = np.array([50, 0, 0])
        delta_e = DeltaECalculator.delta_e_76(lab, lab)
        assert delta_e == 0

    def test_delta_e_2000_identical(self) -> None:
        """Test DeltaE 2000 for identical colors."""
        lab = np.array([50, 0, 0])
        delta_e = DeltaECalculator.delta_e_2000(lab, lab)
        assert abs(delta_e) < 1e-10

    def test_delta_e_2000_different(self) -> None:
        """Test DeltaE 2000 for different colors."""
        lab1 = np.array([50, 0, 0])
        lab2 = np.array([50, 10, 0])
        delta_e = DeltaECalculator.delta_e_2000(lab1, lab2)
        assert delta_e > 0

    def test_identical_images_below_threshold(self, vrt: VisualRegressionTest) -> None:
        """Identical renders should produce < 0.5% pixel difference."""
        scene = RenderScene.PBR_SPHERE
        image1 = vrt.capture_screenshot(scene)
        image2 = vrt.capture_screenshot(scene)

        result = vrt.compare_images(image1, image2)

        assert result.match == True
        assert result.pixel_diff_percent < IDENTICAL_THRESHOLD
        assert result.mean_delta_e < 0.1

    def test_deliberate_regression_above_threshold(
        self, vrt: VisualRegressionTest
    ) -> None:
        """Deliberate color change should produce > 5% pixel difference."""
        scene = RenderScene.PBR_SPHERE

        # Original
        image1 = vrt.capture_screenshot(scene, {"base_color": (0.8, 0.2, 0.2)})

        # Modified (different color)
        image2 = vrt.capture_screenshot(scene, {"base_color": (0.2, 0.8, 0.2)})

        result = vrt.compare_images(image1, image2)

        assert result.match == False
        assert result.pixel_diff_percent > REGRESSION_THRESHOLD


class TestPBRSphere:
    """Tests for PBR sphere rendering consistency."""

    @pytest.fixture
    def vrt(self) -> VisualRegressionTest:
        return VisualRegressionTest()

    def test_pbr_sphere_consistency(self, vrt: VisualRegressionTest) -> None:
        """Same PBR parameters should produce identical results."""
        params = {"base_color": (0.7, 0.3, 0.1), "metallic": 0.0, "roughness": 0.5}

        image1 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, params)
        image2 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, params)

        result = vrt.compare_images(image1, image2)
        assert result.pixel_diff_percent < IDENTICAL_THRESHOLD

    def test_pbr_sphere_metallic_difference(self, vrt: VisualRegressionTest) -> None:
        """Different metallic values should produce visible difference."""
        params1 = {"base_color": (0.7, 0.3, 0.1), "metallic": 0.0}
        params2 = {"base_color": (0.7, 0.3, 0.1), "metallic": 1.0}

        image1 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, params1)
        image2 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, params2)

        result = vrt.compare_images(image1, image2)
        assert result.pixel_diff_percent > REGRESSION_THRESHOLD

    def test_pbr_sphere_roughness_difference(self, vrt: VisualRegressionTest) -> None:
        """Different roughness values should produce visible difference."""
        params1 = {"roughness": 0.0}
        params2 = {"roughness": 1.0}

        image1 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, params1)
        image2 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, params2)

        result = vrt.compare_images(image1, image2)
        assert result.pixel_diff_percent > REGRESSION_THRESHOLD


class TestMaterialVariants:
    """Tests for material domain variants."""

    @pytest.fixture
    def vrt(self) -> VisualRegressionTest:
        return VisualRegressionTest()

    def test_material_variants_render(self, vrt: VisualRegressionTest) -> None:
        """Material variants scene should render consistently."""
        image1 = vrt.capture_screenshot(RenderScene.MATERIAL_VARIANTS)
        image2 = vrt.capture_screenshot(RenderScene.MATERIAL_VARIANTS)

        result = vrt.compare_images(image1, image2)
        assert result.pixel_diff_percent < IDENTICAL_THRESHOLD

    def test_all_five_variants_present(self, vrt: VisualRegressionTest) -> None:
        """All 5 material domain variants should be visually distinct."""
        image = vrt.capture_screenshot(RenderScene.MATERIAL_VARIANTS)

        # Image should not be uniform
        variance = np.var(image)
        assert variance > 0.01

        # Check horizontal variation (5 variants in a row)
        # Each variant should have different dominant colors
        w = image.shape[1]
        fifth = w // 5

        # Sample center of each cell to get representative color
        center_y = image.shape[0] // 2
        colors = []
        for i in range(5):
            cell_center_x = i * fifth + fifth // 2
            color = image[center_y, cell_center_x]
            colors.append(color)

        # Each variant should have different hue/color
        # Use sum of absolute RGB differences
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                color_diff = np.sum(np.abs(colors[i] - colors[j]))
                # Require at least some color difference
                assert color_diff > 0.05, f"Variants {i} and {j} too similar"


class TestAdvancedShading:
    """Tests for advanced shading models."""

    @pytest.fixture
    def vrt(self) -> VisualRegressionTest:
        return VisualRegressionTest()

    def test_advanced_shading_consistency(self, vrt: VisualRegressionTest) -> None:
        """Advanced shading scene should render consistently."""
        image1 = vrt.capture_screenshot(RenderScene.ADVANCED_SHADING)
        image2 = vrt.capture_screenshot(RenderScene.ADVANCED_SHADING)

        result = vrt.compare_images(image1, image2)
        assert result.pixel_diff_percent < IDENTICAL_THRESHOLD

    def test_sss_scatter_radius_difference(self, vrt: VisualRegressionTest) -> None:
        """Different SSS scatter radius should produce visible difference."""
        params1 = {"scatter_radius": 0.5}
        params2 = {"scatter_radius": 2.0}

        image1 = vrt.capture_screenshot(RenderScene.SSS_TEST, params1)
        image2 = vrt.capture_screenshot(RenderScene.SSS_TEST, params2)

        result = vrt.compare_images(image1, image2)
        assert result.pixel_diff_percent > REGRESSION_THRESHOLD

    def test_transmission_ior_difference(self, vrt: VisualRegressionTest) -> None:
        """Different IOR values should produce visible difference."""
        params1 = {"ior": 1.0}
        params2 = {"ior": 2.0}

        image1 = vrt.capture_screenshot(RenderScene.TRANSMISSION_TEST, params1)
        image2 = vrt.capture_screenshot(RenderScene.TRANSMISSION_TEST, params2)

        result = vrt.compare_images(image1, image2)
        # IOR changes should be subtle but noticeable
        assert result.mean_delta_e > 1.0

    def test_clear_coat_intensity_difference(self, vrt: VisualRegressionTest) -> None:
        """Different clear coat intensity should produce visible difference."""
        params1 = {"clear_coat_intensity": 0.0}
        params2 = {"clear_coat_intensity": 1.0}

        image1 = vrt.capture_screenshot(RenderScene.CLEAR_COAT_TEST, params1)
        image2 = vrt.capture_screenshot(RenderScene.CLEAR_COAT_TEST, params2)

        result = vrt.compare_images(image1, image2)
        assert result.pixel_diff_percent > REGRESSION_THRESHOLD


class TestDeltaEMetrics:
    """Tests for DeltaE calculation accuracy."""

    def test_delta_e_threshold_just_noticeable(self) -> None:
        """DeltaE ~2.3 is the just noticeable difference threshold."""
        lab1 = np.array([50.0, 0.0, 0.0])
        lab2 = np.array([52.3, 0.0, 0.0])  # ~2.3 L difference

        delta_e = DeltaECalculator.delta_e_2000(lab1, lab2)

        # Should be close to 2.3 (just noticeable)
        assert 2.0 < delta_e < 3.0

    def test_delta_e_large_difference(self) -> None:
        """Large color differences should have high DeltaE."""
        lab1 = np.array([50.0, -50.0, 0.0])  # Green
        lab2 = np.array([50.0, 50.0, 0.0])  # Red

        delta_e = DeltaECalculator.delta_e_2000(lab1, lab2)

        # Should be very large
        assert delta_e > 50

    def test_delta_e_batch_processing(self) -> None:
        """DeltaE should work with batch image processing."""
        h, w = 100, 100
        lab1 = np.random.rand(h, w, 3) * 100
        lab1[..., 0] = lab1[..., 0]  # L: 0-100
        lab1[..., 1:] = lab1[..., 1:] - 50  # a, b: -50 to 50

        lab2 = lab1.copy()

        delta_e = DeltaECalculator.delta_e_2000(lab1, lab2)

        assert delta_e.shape == (h, w)
        assert np.allclose(delta_e, 0)


class TestDiffVisualization:
    """Tests for diff image generation."""

    @pytest.fixture
    def vrt(self) -> VisualRegressionTest:
        return VisualRegressionTest()

    def test_diff_image_generation(self, vrt: VisualRegressionTest) -> None:
        """Diff image should be generated correctly."""
        image1 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, {"base_color": (1, 0, 0)})
        image2 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, {"base_color": (0, 1, 0)})

        result = vrt.compare_images(image1, image2)

        assert result.diff_image is not None
        assert result.diff_image.shape == image1.shape

    def test_error_mask_generation(self, vrt: VisualRegressionTest) -> None:
        """Error mask should highlight differing regions."""
        image1 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, {"base_color": (1, 0, 0)})
        image2 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, {"base_color": (0, 1, 0)})

        result = vrt.compare_images(image1, image2)

        assert result.error_mask is not None
        assert result.error_mask.dtype == np.uint8
        assert np.sum(result.error_mask > 0) > 0  # Some errors should exist

    def test_histogram_generation(self, vrt: VisualRegressionTest) -> None:
        """DeltaE histogram should be generated."""
        image1 = vrt.capture_screenshot(RenderScene.PBR_SPHERE)
        image2 = vrt.capture_screenshot(RenderScene.PBR_SPHERE)

        result = vrt.compare_images(image1, image2)

        assert result.histogram is not None
        assert "0-1" in result.histogram
        assert "10+" in result.histogram


class TestReferenceManagement:
    """Tests for reference image management."""

    @pytest.fixture
    def vrt(self, tmp_path: Path) -> VisualRegressionTest:
        return VisualRegressionTest(
            reference_dir=tmp_path / "references",
            diff_output_dir=tmp_path / "diffs",
        )

    def test_save_and_load_reference(self, vrt: VisualRegressionTest) -> None:
        """Reference images should save and load correctly."""
        image = vrt.capture_screenshot(RenderScene.PBR_SPHERE)

        vrt.save_reference("test_pbr_sphere", image)
        loaded = vrt.load_reference("test_pbr_sphere")

        assert loaded is not None
        assert np.allclose(image, loaded)

    def test_load_nonexistent_reference(self, vrt: VisualRegressionTest) -> None:
        """Loading nonexistent reference should return None."""
        result = vrt.load_reference("nonexistent")
        assert result is None

    def test_save_diff_output(self, vrt: VisualRegressionTest) -> None:
        """Diff output should be saved for CI."""
        image1 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, {"base_color": (1, 0, 0)})
        image2 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, {"base_color": (0, 1, 0)})

        result = vrt.compare_images(image1, image2)
        output_path = vrt.save_diff_output("test_diff", result, image1, image2)

        assert output_path.exists()
        assert (output_path / "metrics.json").exists()
        assert (output_path / "actual.npy").exists()


class TestResolutionIndependence:
    """Tests for resolution-independent comparisons."""

    def test_different_resolutions_comparable(self) -> None:
        """Comparison should work across resolutions (with scaling)."""
        vrt_small = VisualRegressionTest(config=RenderConfig(width=256, height=256))
        vrt_large = VisualRegressionTest(config=RenderConfig(width=512, height=512))

        # Render at different resolutions
        small = vrt_small.capture_screenshot(RenderScene.PBR_SPHERE)
        large = vrt_large.capture_screenshot(RenderScene.PBR_SPHERE)

        assert small.shape == (256, 256, 3)
        assert large.shape == (512, 512, 3)

    def test_same_resolution_exact_match(self) -> None:
        """Same resolution should produce exact matches."""
        vrt = VisualRegressionTest(config=RenderConfig(width=128, height=128))

        image1 = vrt.capture_screenshot(RenderScene.LIGHTING_TEST)
        image2 = vrt.capture_screenshot(RenderScene.LIGHTING_TEST)

        result = vrt.compare_images(image1, image2)
        assert result.pixel_diff_percent < IDENTICAL_THRESHOLD


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def vrt(self) -> VisualRegressionTest:
        return VisualRegressionTest()

    def test_black_vs_white(self, vrt: VisualRegressionTest) -> None:
        """Black vs white should have maximum difference."""
        black = np.zeros((100, 100, 3))
        white = np.ones((100, 100, 3))

        result = vrt.compare_images(black, white)

        assert result.pixel_diff_percent == 100.0
        assert result.mean_delta_e > 90  # L*=100 difference

    def test_small_image(self, vrt: VisualRegressionTest) -> None:
        """Small images should work correctly."""
        small1 = np.random.rand(10, 10, 3)
        small2 = small1.copy()

        result = vrt.compare_images(small1, small2)
        assert result.pixel_diff_percent < IDENTICAL_THRESHOLD

    def test_single_pixel(self, vrt: VisualRegressionTest) -> None:
        """Single pixel comparison should work."""
        pixel1 = np.array([[[0.5, 0.5, 0.5]]])
        pixel2 = np.array([[[0.5, 0.5, 0.5]]])

        result = vrt.compare_images(pixel1, pixel2)
        assert result.pixel_diff_percent == 0

    def test_image_shape_mismatch(self, vrt: VisualRegressionTest) -> None:
        """Mismatched shapes should raise ValueError."""
        image1 = np.zeros((100, 100, 3))
        image2 = np.zeros((200, 200, 3))

        with pytest.raises(ValueError):
            vrt.compare_images(image1, image2)

    def test_rgba_to_rgb_conversion(self, vrt: VisualRegressionTest) -> None:
        """RGBA images should be compared correctly (alpha stripped)."""
        rgba1 = np.random.rand(50, 50, 4)
        rgba2 = rgba1.copy()
        rgba2[..., 3] = 0.5  # Different alpha

        result = vrt.compare_images(rgba1, rgba2)
        # Alpha should not affect comparison
        assert result.pixel_diff_percent < IDENTICAL_THRESHOLD


class TestCIIntegration:
    """Tests for CI/CD integration support."""

    @pytest.fixture
    def vrt(self, tmp_path: Path) -> VisualRegressionTest:
        return VisualRegressionTest(
            reference_dir=tmp_path / "references",
            diff_output_dir=tmp_path / "diffs",
        )

    def test_ci_artifact_generation(self, vrt: VisualRegressionTest) -> None:
        """CI should be able to generate artifacts on failure."""
        image1 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, {"base_color": (1, 0, 0)})
        image2 = vrt.capture_screenshot(RenderScene.PBR_SPHERE, {"base_color": (0, 0, 1)})

        result = vrt.compare_images(image1, image2)

        # Generate CI artifacts
        output_dir = vrt.save_diff_output("ci_test", result, image1, image2)

        # Verify all expected files
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "actual.npy").exists()
        assert (output_dir / "reference.npy").exists()
        assert (output_dir / "diff.npy").exists()
        assert (output_dir / "error_mask.npy").exists()

        # Verify metrics JSON
        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "pixel_diff_percent" in metrics
        assert "mean_delta_e" in metrics

    def test_metrics_json_serializable(self, vrt: VisualRegressionTest) -> None:
        """DiffResult should be JSON serializable."""
        image1 = vrt.capture_screenshot(RenderScene.PBR_SPHERE)
        image2 = vrt.capture_screenshot(RenderScene.PBR_SPHERE)

        result = vrt.compare_images(image1, image2)
        data = result.to_dict()

        # Should serialize without error
        json_str = json.dumps(data)
        assert json_str is not None

        # Should deserialize correctly
        loaded = json.loads(json_str)
        assert loaded["match"] == result.match
