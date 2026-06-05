"""Visual regression testing for TRINITY rendering subsystem.

This module provides screenshot comparison testing with:
- DeltaE perceptual difference metric (CIE LAB color space)
- Per-pixel error visualization
- Reference image management
- CI integration support

Acceptance criteria:
- Identical renders: < 0.5% pixel difference
- Deliberate regression: > 5% pixel difference
"""

from .test_visual_regression import (
    DiffResult,
    VisualRegressionTest,
    RenderScene,
    MaterialVariant,
)

__all__ = [
    "DiffResult",
    "VisualRegressionTest",
    "RenderScene",
    "MaterialVariant",
]
