"""
Prediction module for client-side prediction and server reconciliation.

This module provides systems for:
- Client-side input prediction
- Server state reconciliation
- Entity interpolation for remote entities
- Smoothing corrections to hide network artifacts
"""

from engine.networking.prediction.client_prediction import (
    InputBuffer,
    PredictionState,
    ClientPredictor,
    BufferedInput,
)
from engine.networking.prediction.server_reconciliation import (
    ReconciliationResult,
    ServerReconciler,
    ReconciliationConfig,
    ReconciliationStats,
    ReconciliationHistory,
    ReconciliationFrame,
)
from engine.networking.prediction.entity_interpolation import (
    Snapshot,
    InterpolationBuffer,
    InterpolationMode,
    InterpolatedState,
    EntityInterpolator,
    lerp_position,
    slerp_rotation,
    hermite_interpolate,
)
from engine.networking.prediction.smoothing import (
    SmoothingMethod,
    CorrectionSmoother,
    SmoothingConfig,
    CorrectionState,
    VisualSmoother,
    smooth_position,
    smooth_rotation,
    exponential_smooth,
    exponential_smooth_vector,
)

__all__ = [
    # Client prediction
    "InputBuffer",
    "PredictionState",
    "ClientPredictor",
    "BufferedInput",
    # Server reconciliation
    "ReconciliationResult",
    "ServerReconciler",
    "ReconciliationConfig",
    "ReconciliationStats",
    "ReconciliationHistory",
    "ReconciliationFrame",
    # Entity interpolation
    "Snapshot",
    "InterpolationBuffer",
    "InterpolationMode",
    "InterpolatedState",
    "EntityInterpolator",
    "lerp_position",
    "slerp_rotation",
    "hermite_interpolate",
    # Smoothing
    "SmoothingMethod",
    "CorrectionSmoother",
    "SmoothingConfig",
    "CorrectionState",
    "VisualSmoother",
    "smooth_position",
    "smooth_rotation",
    "exponential_smooth",
    "exponential_smooth_vector",
]
