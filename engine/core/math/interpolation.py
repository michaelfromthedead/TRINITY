"""Interpolation functions and easing curves."""

from __future__ import annotations

import math

from engine.core.constants import MATH_EPSILON_TIGHT


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def inverse_lerp(a: float, b: float, value: float) -> float:
    if abs(b - a) < MATH_EPSILON_TIGHT:
        return 0.0
    return (value - a) / (b - a)


def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    t = inverse_lerp(in_min, in_max, value)
    return lerp(out_min, out_max, t)


def clamp(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, value))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0) if edge1 != edge0 else 0.0, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0) if edge1 != edge0 else 0.0, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# Easing functions (t in [0, 1])

def in_quad(t: float) -> float:
    return t * t


def out_quad(t: float) -> float:
    return t * (2.0 - t)


def in_out_quad(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return -1.0 + (4.0 - 2.0 * t) * t


def in_cubic(t: float) -> float:
    return t * t * t


def out_cubic(t: float) -> float:
    u = t - 1.0
    return u * u * u + 1.0


def in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t * t * t
    u = 2.0 * t - 2.0
    return 0.5 * u * u * u + 1.0


class SpringDamper:
    """Critically-damped spring for smooth interpolation."""
    __slots__ = ("position", "velocity", "target", "omega")

    def __init__(self, position: float = 0.0, velocity: float = 0.0,
                 target: float = 0.0, omega: float = 10.0) -> None:
        self.position = position
        self.velocity = velocity
        self.target = target
        self.omega = omega

    def update(self, dt: float) -> float:
        """Advance the spring by dt seconds. Returns new position."""
        if dt < 0.0:
            raise ValueError(f"SpringDamper.update: dt must be non-negative, got {dt}")
        if dt == 0.0:
            return self.position
        delta = self.position - self.target
        exp_term = math.exp(-self.omega * dt)
        self.position = self.target + (delta + (self.velocity + self.omega * delta) * dt) * exp_term
        self.velocity = (self.velocity - self.omega * (self.velocity + self.omega * delta) * dt) * exp_term
        return self.position
