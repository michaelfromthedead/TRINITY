"""
Whitebox tests for sdCone(p, h, r1, r2) WGSL function (T-DEMO-1.5).

Tests a Python model implementation matching WGSL semantics, verifying:
  - Internal ca/cb formula decomposition
  - h=0 guard clause returning max(length(p.xz) - max(r1, r2), abs(p.y))
  - Centered cylinder case (r1 = r2)
  - Pointed cone (r1 = 0 or r2 = 0)
  - Inside cone distances (negative)
  - On-surface distances (zero)
  - Above/below cap detection (positive outside)
  - Radial symmetry around y-axis

WHITEBOX coverage plan:
  Path A: Guard clause (h < 1e-8) returns flat-disc fallback
  Path B: ca decomposition — cap distance components ca.x and ca.y
  Path C: cb decomposition — side distance with clamped projection
  Path D: Centered cylinder (r1=r2) — matches capped cylinder behavior
  Path E: Pointed cone (r1=0) — apex at y=0, base at y=h
  Path F: Inverted cone (r2=0) — base at y=0, apex at y=h
  Path G: Inside cone — negative signed distance on axis
  Path H: On side surface — distance ~0 for various heights
  Path I: Above top cap — y > h returns positive distance
  Path J: Below bottom cap — y < 0 returns positive distance
  Path K: Radial symmetry — any rotation around y yields same distance

Reference: Inigo Quilez -- SDF Primitives: sdCappedCone
https://iquilezles.org/articles/distfunctions/
"""

import math

import pytest

# =============================================================================
# Python model implementation matching WGSL semantics
# =============================================================================

TOL = 1e-6


def py_sdCone(p, h, r1, r2):
    """Model of WGSL sdCone(p, h, r1, r2).

    WGSL semantics (from sdf_cone.wgsl):
      if (h < 1e-8) {
          return max(length(p.xz) - max(r1, r2), abs(p.y));
      }
      let half_h = h * 0.5;
      let q = vec2<f32>(length(p.xz), p.y - half_h);
      let k1 = vec2<f32>(r2, half_h);
      let k2 = vec2<f32>(r2 - r1, h);
      let ca = vec2<f32>(
          q.x - min(q.x, select(r2, r1, q.y < 0.0)),
          abs(q.y) - half_h
      );
      let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
      let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
      return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
    """
    px, py, pz = p

    # Guard clause: degenerate cone (zero height)
    if h < 1e-8:
        xz_len = math.sqrt(px * px + pz * pz)
        return max(xz_len - max(r1, r2), abs(py))

    half_h = h * 0.5

    # 2D cross-section: radial distance on xz, centered height on y
    qx = math.sqrt(px * px + pz * pz)
    qy = py - half_h

    # IQ reference points for the trapezoid in 2D
    k1x, k1y = r2, half_h
    k2x, k2y = r2 - r1, h

    # ---- ca: cap distance ----
    # ca.x: radial excess relative to the nearer cap radius
    r_at_y = r1 if qy < 0.0 else r2
    ca_x = qx - min(qx, r_at_y)
    # ca.y: vertical distance to the nearest cap plane
    ca_y = abs(qy) - half_h

    # ---- cb: side distance ----
    # Clamped projection of (k1 - q) onto the side direction k2
    dot_k1q_k2 = (k1x - qx) * k2x + (k1y - qy) * k2y
    dot_k2_k2 = k2x * k2x + k2y * k2y
    # WGSL clamp handles division by zero gracefully (returns NaN → clamp → 0)
    t = max(0.0, min(1.0, dot_k1q_k2 / dot_k2_k2)) if dot_k2_k2 != 0.0 else 0.0

    cb_x = qx - k1x + k2x * t
    cb_y = qy - k1y + k2y * t

    # ---- sign ----
    # Inside when radially inside the side (cb.x < 0) AND between caps (ca.y < 0)
    s = -1.0 if (cb_x < 0.0 and ca_y < 0.0) else 1.0

    # ---- result ----
    ca_dot = ca_x * ca_x + ca_y * ca_y
    cb_dot = cb_x * cb_x + cb_y * cb_y
    return s * math.sqrt(min(ca_dot, cb_dot))


# =============================================================================
# Helpers
# =============================================================================


def _len_xz(x, z):
    """Radial distance from y-axis."""
    return math.sqrt(x * x + z * z)


def _py_ca(p, h, r1, r2):
    """Expose ca decomposition for whitebox verification."""
    px, py, pz = p
    half_h = h * 0.5
    qx = math.sqrt(px * px + pz * pz)
    qy = py - half_h
    r_at_y = r1 if qy < 0.0 else r2
    ca_x = qx - min(qx, r_at_y)
    ca_y = abs(qy) - half_h
    return (ca_x, ca_y)


def _py_cb(p, h, r1, r2):
    """Expose cb decomposition for whitebox verification."""
    px, py, pz = p
    half_h = h * 0.5
    qx = math.sqrt(px * px + pz * pz)
    qy = py - half_h
    k1x, k1y = r2, half_h
    k2x, k2y = r2 - r1, h

    dot_k1q_k2 = (k1x - qx) * k2x + (k1y - qy) * k2y
    dot_k2_k2 = k2x * k2x + k2y * k2y
    t = max(0.0, min(1.0, dot_k1q_k2 / dot_k2_k2)) if dot_k2_k2 != 0.0 else 0.0

    cb_x = qx - k1x + k2x * t
    cb_y = qy - k1y + k2y * t
    return (cb_x, cb_y)


# =============================================================================
# Test: T-DEMO-1.5 Formula Decomposition — ca (cap distance)
# =============================================================================


class TestCapDecomposition:
    """Whitebox tests for ca = (q.x - min(q.x, r_at_y), abs(q.y) - half_h)."""

    def test_ca_radial_above_below_select(self):
        """ca.x uses select(r2, r1, q.y < 0): below center picks r1, above picks r2."""
        h, half_h = 2.0, 1.0
        r1, r2 = 1.0, 3.0
        # q.y < 0 (below center): r_at_y = r1 = 1
        ca = _py_ca((2.0, 0.0, 0.0), h, r1, r2)  # q = (2, -1)
        # ca_x = qx - min(qx, r1) = 2 - min(2, 1) = 2 - 1 = 1
        assert ca[0] == pytest.approx(1.0, abs=TOL)
        # q.y >= 0 (above center): r_at_y = r2 = 3
        ca2 = _py_ca((2.0, 2.0, 0.0), h, r1, r2)  # q = (2, 1)
        # ca_x = 2 - min(2, 3) = 2 - 2 = 0 (inside radius)
        assert ca2[0] == pytest.approx(0.0, abs=TOL)

    def test_ca_radial_inside_clamps_to_zero(self):
        """ca.x = 0 when q.x <= r_at_y (radially inside the cap)."""
        h = 2.0
        # q.x = 0.5, r_at_y = 1 (below center) → 0.5 - min(0.5, 1) = 0
        ca = _py_ca((0.5, 0.0, 0.0), h, 1.0, 2.0)
        assert ca[0] == pytest.approx(0.0, abs=TOL)

    def test_ca_vertical_between_caps(self):
        """ca.y = abs(q.y) - half_h: negative when between cap planes."""
        h, half_h = 2.0, 1.0
        # q at center (y=1, shifted to 0): ca_y = |0| - 1 = -1
        ca = _py_ca((0.0, 1.0, 0.0), h, 1.0, 2.0)
        assert ca[1] == pytest.approx(-1.0, abs=TOL)
        # q at mid (y=0.5, shifted to -0.5): ca_y = |-0.5| - 1 = -0.5
        ca2 = _py_ca((0.0, 0.5, 0.0), h, 1.0, 2.0)
        assert ca2[1] == pytest.approx(-0.5, abs=TOL)

    def test_ca_vertical_on_cap(self):
        """ca.y = 0 when q.y is exactly at a cap plane."""
        h, half_h = 2.0, 1.0
        # Bottom cap: y=0, shifted to qy=-1 → ca_y = |-1| - 1 = 0
        ca_bottom = _py_ca((0.0, 0.0, 0.0), h, 1.0, 2.0)
        assert ca_bottom[1] == pytest.approx(0.0, abs=TOL)
        # Top cap: y=2, shifted to qy=1 → ca_y = |1| - 1 = 0
        ca_top = _py_ca((0.0, 2.0, 0.0), h, 1.0, 2.0)
        assert ca_top[1] == pytest.approx(0.0, abs=TOL)

    def test_ca_vertical_outside_caps(self):
        """ca.y > 0 when q.y is beyond a cap plane (outside)."""
        h, half_h = 2.0, 1.0
        # Above top: y=3, shifted to qy=2 → ca_y = |2| - 1 = 1
        ca_above = _py_ca((0.0, 3.0, 0.0), h, 1.0, 2.0)
        assert ca_above[1] == pytest.approx(1.0, abs=TOL)
        # Below bottom: y=-1, shifted to qy=-2 → ca_y = |-2| - 1 = 1
        ca_below = _py_ca((0.0, -1.0, 0.0), h, 1.0, 2.0)
        assert ca_below[1] == pytest.approx(1.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.5 Formula Decomposition — cb (side distance)
# =============================================================================


class TestSideDecomposition:
    """Whitebox tests for cb = q - k1 + k2 * clamp(dot(k1-q, k2)/dot(k2,k2), 0, 1)."""

    def test_cb_cylinder_side_center(self):
        """cb for cylinder (r1=r2) at mid-height: projection lands at midpoint (t=0.5)."""
        h = 2.0
        # Cylinder r1=r2=1, point on side at center: p=(1, 1, 0)
        # q=(1, 0), k1=(1, 1), k2=(0, 2)
        # dot(k1-q, k2) = dot((0,-1), (0,2)) = -2
        # dot(k2,k2) = 4
        # t = clamp(-2/4, 0, 1) = 0
        # Wait, that gives t=0, not 0.5. Let me recalculate.
        # k1 - q = (1, 1) - (1, 0) = (0, 1)
        # dot((0,1), (0,2)) = 0+2 = 2
        # t = 2/4 = 0.5
        cb = _py_cb((1.0, 1.0, 0.0), h, 1.0, 1.0)
        # cb = (1, 0) - (1, 1) + (0, 2)*0.5 = (0, -1) + (0, 1) = (0, 0)
        assert cb[0] == pytest.approx(0.0, abs=TOL)
        assert cb[1] == pytest.approx(0.0, abs=TOL)

    def test_cb_cylinder_inside_center(self):
        """cb for cylinder at axis center: radially inside (cb.x < 0)."""
        h = 2.0
        # Center at p=(0, 1, 0): q=(0, 0)
        # k1=(1, 1), k2=(0, 2), k1-q=(1, 1)
        # dot((1,1),(0,2)) = 2
        # t = 2/4 = 0.5
        # cb = (0,0)-(1,1)+(0,2)*0.5 = (-1,-1)+(0,1) = (-1,0)
        cb = _py_cb((0.0, 1.0, 0.0), h, 1.0, 1.0)
        assert cb[0] == pytest.approx(-1.0, abs=TOL)
        assert cb[1] == pytest.approx(0.0, abs=TOL)

    def test_cb_clamp_below_zero(self):
        """Projection t clamps to 0 when dot(k1-q, k2) < 0."""
        h = 2.0
        # Above the cone: p=(0, 5, 0), q=(0, 4)
        # k1=(1, 1), k2=(0, 2), k1-q=(1, -3)
        # dot((1,-3),(0,2)) = -6
        # t = clamp(-6/4) = 0
        # cb = (0,4)-(1,1)+(0,2)*0 = (-1, 3)
        cb = _py_cb((0.0, 5.0, 0.0), h, 1.0, 1.0)
        assert cb[0] == pytest.approx(-1.0, abs=TOL)
        assert cb[1] == pytest.approx(3.0, abs=TOL)

    def test_cb_clamp_above_one(self):
        """Projection t clamps to 1 when dot(k1-q, k2) > dot(k2, k2)."""
        h = 2.0
        # Below the cone: p=(0, -1, 0), q=(0, -2)
        # k1=(1, 1), k2=(0, 2), k1-q=(1, 3)
        # dot((1,3),(0,2)) = 6
        # t = clamp(6/4, 0, 1) = 1
        # cb = (0,-2)-(1,1)+(0,2)*1 = (-1,-3)+(0,2) = (-1,-1)
        cb = _py_cb((0.0, -1.0, 0.0), h, 1.0, 1.0)
        assert cb[0] == pytest.approx(-1.0, abs=TOL)
        assert cb[1] == pytest.approx(-1.0, abs=TOL)

    def test_cb_pointed_cone_side(self):
        """cb for pointed cone (r1=0) on side surface: cb = (0, 0)."""
        h = 1.0
        # Pointed cone r1=0, r2=1, h=1
        # Side at y=0.5, radius=0.5: p=(0.5, 0.5, 0)
        # half_h=0.5, q=(0.5, 0)
        # k1=(1, 0.5), k2=(1, 1), k1-q=(0.5, 0.5)
        # dot((0.5,0.5),(1,1)) = 1, dot(k2,k2)=2, t=0.5
        # cb = (0.5,0)-(1,0.5)+(1,1)*0.5 = (-0.5,-0.5)+(0.5,0.5) = (0,0)
        cb = _py_cb((0.5, 0.5, 0.0), h, 0.0, 1.0)
        assert cb[0] == pytest.approx(0.0, abs=TOL)
        assert cb[1] == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.5 Guard Clause (h=0)
# =============================================================================


class TestGuardClause:
    """Tests for the degenerate cone guard: h < 1e-8."""

    def test_guard_returns_disc_max_radius(self):
        """h=0 collapses to a flat disc: max(length(xz) - max(r1,r2), abs(y))."""
        # Disc radius = max(1, 2) = 2. Point (2, 0, 0) is on the disc edge.
        assert py_sdCone((2.0, 0.0, 0.0), 0.0, 1.0, 2.0) == pytest.approx(0.0, abs=TOL)

    def test_guard_outside_disc(self):
        """Point beyond the disc radius: distance = radial excess."""
        # Disc radius = max(1, 2) = 2. Point (3, 0, 0) is 1 unit outside.
        assert py_sdCone((3.0, 0.0, 0.0), 0.0, 1.0, 2.0) == pytest.approx(1.0, abs=TOL)

    def test_guard_above_disc(self):
        """Point above the disc: distance = y-offset."""
        # Disc at y=0, radius=2. Point (0, 1, 0) is 1 unit above.
        assert py_sdCone((0.0, 1.0, 0.0), 0.0, 1.0, 2.0) == pytest.approx(1.0, abs=TOL)

    def test_guard_below_disc(self):
        """Point below the disc: distance = |y|."""
        # Disc at y=0, radius=2. Point (0, -1, 0) is 1 unit below.
        assert py_sdCone((0.0, -1.0, 0.0), 0.0, 1.0, 2.0) == pytest.approx(1.0, abs=TOL)

    def test_guard_inside_disc_on_plane(self):
        """Point inside the disc on the plane: max(negative, 0) = 0."""
        assert py_sdCone((0.0, 0.0, 0.0), 0.0, 1.0, 2.0) == pytest.approx(0.0, abs=TOL)

    def test_guard_r1_larger(self):
        """When r1 > r2, max(r1, r2) uses r1 as disc radius."""
        # Disc radius = max(3, 1) = 3. Point (3, 0, 0) is on edge.
        assert py_sdCone((3.0, 0.0, 0.0), 0.0, 3.0, 1.0) == pytest.approx(0.0, abs=TOL)

    def test_guard_negative_h_triggers(self):
        """Negative h also triggers the guard (h < 1e-8).

        Disc radius = max(1, 1) = 1. Point (-2, 0, 0) is 1 unit outside
        the disc edge, so distance = length((-2,0)) - 1 = 2 - 1 = 1.
        """
        assert py_sdCone((-2.0, 0.0, 0.0), -1.0, 1.0, 1.0) == pytest.approx(1.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.5 Centered Cone — Cylinder (r1 = r2)
# =============================================================================


class TestCylinderCase:
    """Cylinder: r1 = r2 = 1, h = 2. Matches capped cylinder behavior."""

    H = 2.0
    R = 1.0

    def test_center_returns_negative(self):
        """Center of cylinder at mid-height (y=1) returns distance -1."""
        d = py_sdCone((0.0, 1.0, 0.0), self.H, self.R, self.R)
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_side_surface_zero(self):
        """On the cylinder side at mid-height: distance 0."""
        assert py_sdCone((self.R, 1.0, 0.0), self.H, self.R, self.R) == pytest.approx(0.0, abs=TOL)

    def test_top_cap_surface_zero(self):
        """On the top cap center: distance 0."""
        assert py_sdCone((0.0, self.H, 0.0), self.H, self.R, self.R) == pytest.approx(0.0, abs=TOL)

    def test_bottom_cap_surface_zero(self):
        """On the bottom cap center: distance 0."""
        assert py_sdCone((0.0, 0.0, 0.0), self.H, self.R, self.R) == pytest.approx(0.0, abs=TOL)

    def test_outside_radially_positive(self):
        """Outside the cylinder radially: distance = radial excess."""
        d = py_sdCone((3.0, 1.0, 0.0), self.H, self.R, self.R)
        assert d == pytest.approx(2.0, abs=TOL)

    def test_outside_above_cap(self):
        """Above the top cap: distance = y-distance to top cap."""
        d = py_sdCone((0.0, 4.0, 0.0), self.H, self.R, self.R)
        assert d == pytest.approx(2.0, abs=TOL)

    def test_outside_below_cap(self):
        """Below the bottom cap: distance = |y-distance| to bottom cap."""
        d = py_sdCone((0.0, -2.0, 0.0), self.H, self.R, self.R)
        assert d == pytest.approx(2.0, abs=TOL)

    def test_interior_sweep(self):
        """Inside cylinder: distance increases linearly toward the side."""
        prev = py_sdCone((0.0, 1.0, 0.0), self.H, self.R, self.R)
        assert prev == pytest.approx(-1.0, abs=TOL)
        for x in [0.25, 0.5, 0.75]:
            d = py_sdCone((x, 1.0, 0.0), self.H, self.R, self.R)
            assert d < 0.0, f"Inside point should be negative, got {d}"
            assert d >= prev, (
                f"Distance should increase (less negative) toward surface, "
                f"got {d} < {prev} at x={x}"
            )
            prev = d
        # On surface
        assert py_sdCone((self.R, 1.0, 0.0), self.H, self.R, self.R) == pytest.approx(0.0, abs=TOL)

    def test_linear_exterior(self):
        """Outside radially: distance increases by exactly the offset from surface."""
        for offset in [0.5, 1.0, 2.0, 5.0]:
            d = py_sdCone((self.R + offset, 1.0, 0.0), self.H, self.R, self.R)
            assert d == pytest.approx(offset, abs=TOL), (
                f"Expected offset {offset}, got {d}"
            )


# =============================================================================
# Test: T-DEMO-1.5 Pointed Cone (r1 = 0)
# =============================================================================


class TestPointedCone:
    """Pointed cone: r1 = 0, r2 = 1, h = 1. Apex at y=0, base at y=1."""

    H = 1.0
    R1 = 0.0
    R2 = 1.0

    def test_apex_on_surface(self):
        """Apex at y=0 (r1=0): distance ~0."""
        d = py_sdCone((0.0, 0.0, 0.0), self.H, self.R1, self.R2)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_base_center_on_surface(self):
        """Base center at y=1 (r2=1): distance ~0."""
        d = py_sdCone((0.0, self.H, 0.0), self.H, self.R1, self.R2)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_side_mid_height_on_surface(self):
        """Mid-height on the side: radius = r1 + (r2-r1)*0.5 = 0.5 at y=0.5."""
        d = py_sdCone((0.5, 0.5, 0.0), self.H, self.R1, self.R2)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_side_quarter_height_on_surface(self):
        """Quarter-height: radius = 0.25 at y=0.25."""
        d = py_sdCone((0.25, 0.25, 0.0), self.H, self.R1, self.R2)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_side_three_quarter_height_on_surface(self):
        """Three-quarter-height: radius = 0.75 at y=0.75."""
        d = py_sdCone((0.75, 0.75, 0.0), self.H, self.R1, self.R2)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_inside_apex_region(self):
        """Inside the cone near apex: negative distance."""
        d = py_sdCone((0.0, 0.1, 0.0), self.H, self.R1, self.R2)
        assert d < 0.0, f"Inside pointed cone should be negative, got {d}"

    def test_outside_above_apex(self):
        """Below the apex (y < 0): outside."""
        d = py_sdCone((0.0, -0.1, 0.0), self.H, self.R1, self.R2)
        assert d > 0.0, f"Below apex should be positive, got {d}"

    def test_outside_above_base(self):
        """Above the base (y > h): outside."""
        d = py_sdCone((0.0, 1.5, 0.0), self.H, self.R1, self.R2)
        assert d > 0.0, f"Above base should be positive, got {d}"


# =============================================================================
# Test: T-DEMO-1.5 Inverted Cone (r2 = 0)
# =============================================================================


class TestInvertedCone:
    """Inverted cone: r1 = 1, r2 = 0, h = 1. Base at y=0, apex at y=1."""

    H = 1.0
    R1 = 1.0
    R2 = 0.0

    def test_base_center_on_surface(self):
        """Base center at y=0 (r1=1): distance ~0."""
        d = py_sdCone((0.0, 0.0, 0.0), self.H, self.R1, self.R2)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_apex_on_surface(self):
        """Apex at y=h (r2=0): distance ~0."""
        d = py_sdCone((0.0, self.H, 0.0), self.H, self.R1, self.R2)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_side_mid_height_on_surface(self):
        """Mid-height on the side: radius = 1 - (1-0)*0.5 = 0.5 at y=0.5."""
        d = py_sdCone((0.5, 0.5, 0.0), self.H, self.R1, self.R2)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_side_quarter_height_on_surface(self):
        """Quarter-height: radius = 0.75 at y=0.25."""
        d = py_sdCone((0.75, 0.25, 0.0), self.H, self.R1, self.R2)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_outside_below_base(self):
        """Below the base (y < 0): outside."""
        d = py_sdCone((0.0, -0.1, 0.0), self.H, self.R1, self.R2)
        assert d > 0.0, f"Below base should be positive, got {d}"

    def test_outside_above_apex(self):
        """Above the apex (y > h): outside."""
        d = py_sdCone((0.0, 1.5, 0.0), self.H, self.R1, self.R2)
        assert d > 0.0, f"Above apex should be positive, got {d}"


# =============================================================================
# Test: T-DEMO-1.5 Inside Cone — Negative Signed Distance
# =============================================================================


class TestInsideCone:
    """Points inside the cone return negative signed distance."""

    def test_cylinder_axis_inside(self):
        """Inside cylinder on axis: distance = -r1 = -1."""
        d = py_sdCone((0.0, 1.0, 0.0), 2.0, 1.0, 1.0)
        assert d < 0.0
        assert d == pytest.approx(-1.0, abs=TOL)

    def test_pointed_interpolated_inside(self):
        """Inside pointed cone at mid-height axis: negative."""
        d = py_sdCone((0.0, 0.5, 0.0), 1.0, 0.0, 1.0)
        assert d < 0.0, f"Inside pointed cone should be negative, got {d}"

    def test_near_surface_inside(self):
        """Just inside the side surface: small negative."""
        d = py_sdCone((0.9, 1.0, 0.0), 2.0, 1.0, 1.0)
        assert d < 0.0
        assert d == pytest.approx(-0.1, abs=TOL)

    def test_deep_inside_cylinder(self):
        """Deep inside wide cylinder: more negative."""
        d = py_sdCone((0.0, 5.0, 0.0), 10.0, 5.0, 5.0)
        assert d < 0.0
        assert d == pytest.approx(-5.0, abs=TOL)

    def test_inside_thin_cone_apex(self):
        """Inside near apex of thin pointed cone: negative."""
        d = py_sdCone((0.0, 0.1, 0.0), 2.0, 0.0, 0.5)
        assert d < 0.0


# =============================================================================
# Test: T-DEMO-1.5 On Side Surface (Various Heights)
# =============================================================================


class TestOnSideSurface:
    """Points exactly on the cone side surface return distance ~0."""

    @pytest.mark.parametrize("y", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_cylinder_surface_all_heights(self, y):
        """Cylinder: side radius is constant = 1 at all heights."""
        d = py_sdCone((1.0, y, 0.0), 1.0, 1.0, 1.0)
        assert d == pytest.approx(0.0, abs=TOL), (
            f"Cylinder at y={y} should be on surface, got {d}"
        )

    @pytest.mark.parametrize("y, r", [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)])
    def test_pointed_surface_various_heights(self, y, r):
        """Pointed cone (r1=0, r2=1): radius interpolates from 0 to 1."""
        d = py_sdCone((r, y, 0.0), 1.0, 0.0, 1.0)
        assert d == pytest.approx(0.0, abs=TOL), (
            f"Pointed cone at y={y}, r={r} should be on surface, got {d}"
        )

    @pytest.mark.parametrize("y, r", [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 0.0)])
    def test_inverted_surface_various_heights(self, y, r):
        """Inverted cone (r1=1, r2=0): radius interpolates from 1 to 0."""
        d = py_sdCone((r, y, 0.0), 1.0, 1.0, 0.0)
        assert d == pytest.approx(0.0, abs=TOL), (
            f"Inverted cone at y={y}, r={r} should be on surface, got {d}"
        )

    def test_tapered_arbitrary_radii(self):
        """Arbitrary r1=2, r2=0.5, h=3: verify at multiple heights."""
        h, r1, r2 = 3.0, 2.0, 0.5
        # Radius at height y: r(y) = r1 + (r2 - r1) * y/h
        for y in [0.0, 1.0, 2.0, 3.0]:
            r_expected = r1 + (r2 - r1) * y / h
            d = py_sdCone((r_expected, y, 0.0), h, r1, r2)
            assert d == pytest.approx(0.0, abs=TOL), (
                f"Tapered cone at y={y}, r={r_expected} should be on surface, got {d}"
            )


# =============================================================================
# Test: T-DEMO-1.5 Above Cap (y > h)
# =============================================================================


class TestAboveCap:
    """Points with y > h should be outside (positive distance)."""

    def test_above_cylinder_returns_positive(self):
        """Point clearly above cylinder cap: positive distance."""
        d = py_sdCone((0.0, 3.0, 0.0), 2.0, 1.0, 1.0)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_above_pointed_cone(self):
        """Point above pointed cone: positive distance."""
        d = py_sdCone((0.0, 2.0, 0.0), 1.0, 0.0, 1.0)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_just_above_cap(self):
        """Just above the cap: small positive distance."""
        eps = 1e-4
        d = py_sdCone((0.0, 2.0 + eps, 0.0), 2.0, 1.0, 1.0)
        assert d == pytest.approx(eps, abs=1e-3)

    def test_above_off_axis(self):
        """Above the cap at an off-axis position."""
        d = py_sdCone((0.5, 3.0, 0.0), 2.0, 1.0, 1.0)
        assert d > 0.0


# =============================================================================
# Test: T-DEMO-1.5 Below Cap (y < 0)
# =============================================================================


class TestBelowCap:
    """Points with y < 0 should be outside (positive distance)."""

    def test_below_cylinder_returns_positive(self):
        """Point clearly below cylinder cap: positive distance."""
        d = py_sdCone((0.0, -1.0, 0.0), 2.0, 1.0, 1.0)
        assert d == pytest.approx(1.0, abs=TOL)

    def test_below_pointed_cone_apex(self):
        """Point below pointed cone apex: positive distance."""
        d = py_sdCone((0.0, -0.5, 0.0), 1.0, 0.0, 1.0)
        assert d == pytest.approx(0.5, abs=TOL)

    def test_just_below_cap(self):
        """Just below the cap: small positive distance."""
        eps = 1e-4
        d = py_sdCone((0.0, -eps, 0.0), 2.0, 1.0, 1.0)
        assert d == pytest.approx(eps, abs=1e-3)

    def test_below_off_axis(self):
        """Below the cap at an off-axis position."""
        d = py_sdCone((0.5, -1.0, 0.0), 2.0, 1.0, 1.0)
        assert d > 0.0


# =============================================================================
# Test: T-DEMO-1.5 Cylinder Case (r1 = r2) — Additional Verification
# =============================================================================


class TestCylinderEdgeCases:
    """Edge cases for the r1 = r2 cylinder reduction."""

    def test_r1_r2_equal_large_height(self):
        """Cylinder with r1=r2=3, h=10: center should be -3."""
        d = py_sdCone((0.0, 5.0, 0.0), 10.0, 3.0, 3.0)
        assert d == pytest.approx(-3.0, abs=TOL)

    def test_r1_r2_equal_tall_thin(self):
        """Tall thin cylinder: r1=r2=0.1, h=10."""
        d = py_sdCone((0.0, 5.0, 0.0), 10.0, 0.1, 0.1)
        assert d == pytest.approx(-0.1, abs=TOL)
        # Just outside radially
        d_out = py_sdCone((0.2, 5.0, 0.0), 10.0, 0.1, 0.1)
        assert d_out == pytest.approx(0.1, abs=TOL)

    def test_on_corner_edge(self):
        """Cylinder top rim: (r, h, 0) should be on the edge (distance ~0)."""
        d = py_sdCone((1.0, 2.0, 0.0), 2.0, 1.0, 1.0)
        assert d == pytest.approx(0.0, abs=TOL)

    def test_cylinder_bottom_rim(self):
        """Cylinder bottom rim: (r, 0, 0) on the edge."""
        d = py_sdCone((1.0, 0.0, 0.0), 2.0, 1.0, 1.0)
        assert d == pytest.approx(0.0, abs=TOL)


# =============================================================================
# Test: T-DEMO-1.5 Radial Symmetry Around y-axis
# =============================================================================


class TestRadialSymmetry:
    """sdCone depends only on length(p.xz), not on the direction of (x,z)."""

    H = 2.0
    R1 = 1.0
    R2 = 2.0

    def _assert_radially_symmetric(self, y, h, r1, r2):
        """Assert that rotation around y produces the same distance."""
        base = py_sdCone((1.0, y, 0.0), h, r1, r2)
        # 90-degree rotation: (0, y, 1)
        r90 = py_sdCone((0.0, y, 1.0), h, r1, r2)
        assert r90 == pytest.approx(base, abs=TOL), (
            f"90-degree rotation at y={y}: {r90} != {base}"
        )
        # 45-degree rotation: (cos45, y, sin45) = (0.7071, y, 0.7071)
        c45 = 1.0 / math.sqrt(2.0)
        r45 = py_sdCone((c45, y, c45), h, r1, r2)
        assert r45 == pytest.approx(base, abs=TOL), (
            f"45-degree rotation at y={y}: {r45} != {base}"
        )
        # Negative x
        neg_x = py_sdCone((-1.0, y, 0.0), h, r1, r2)
        assert neg_x == pytest.approx(base, abs=TOL), (
            f"Negative x at y={y}: {neg_x} != {base}"
        )

    def test_symmetry_at_center(self):
        """Symmetry holds at the cone center (mid-height)."""
        self._assert_radially_symmetric(1.0, self.H, self.R1, self.R2)

    def test_symmetry_at_bottom_cap(self):
        """Symmetry holds at the bottom cap."""
        self._assert_radially_symmetric(0.0, self.H, self.R1, self.R2)

    def test_symmetry_at_top_cap(self):
        """Symmetry holds at the top cap."""
        self._assert_radially_symmetric(self.H, self.H, self.R1, self.R2)

    def test_symmetry_above_cone(self):
        """Symmetry holds above the cone."""
        self._assert_radially_symmetric(3.0, self.H, self.R1, self.R2)

    def test_symmetry_below_cone(self):
        """Symmetry holds below the cone."""
        self._assert_radially_symmetric(-1.0, self.H, self.R1, self.R2)

    def test_symmetry_pointed(self):
        """Symmetry holds for pointed cones."""
        for y in [0.0, 0.5, 1.0]:
            self._assert_radially_symmetric(y, 1.0, 0.0, 1.0)

    def test_symmetry_cylinder(self):
        """Symmetry holds for cylinders (r1=r2)."""
        for y in [0.0, 1.0, 2.0]:
            self._assert_radially_symmetric(y, 2.0, 1.0, 1.0)

    def test_exact_angles_produce_same_result(self):
        """Multiple specific angles produce the same signed distance."""
        h, r1, r2 = 2.0, 1.0, 2.0
        y = 1.0
        radius = 1.5
        results = []
        for angle_deg in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            rad = math.radians(angle_deg)
            x = radius * math.cos(rad)
            z = radius * math.sin(rad)
            d = py_sdCone((x, y, z), h, r1, r2)
            results.append(d)
        # All results should match
        for d in results[1:]:
            assert d == pytest.approx(results[0], abs=TOL), (
                f"Radial symmetry broken: {d} != {results[0]}"
            )


# =============================================================================
# Test: T-DEMO-1.5 Determinism
# =============================================================================


class TestDeterminism:
    """sdCone must be deterministic: same inputs produce same output."""

    def test_repeated_calls_match(self):
        """Repeated calls with the same arguments produce identical results."""
        args = ((0.0, 1.0, 0.0), 2.0, 1.0, 1.0)
        first = py_sdCone(*args)
        for _ in range(20):
            assert py_sdCone(*args) == pytest.approx(first, abs=TOL)

    def test_pointed_cone_deterministic(self):
        """Pointed cone is also deterministic."""
        args = ((0.3, 0.3, 0.0), 1.0, 0.0, 1.0)
        first = py_sdCone(*args)
        for _ in range(10):
            assert py_sdCone(*args) == pytest.approx(first, abs=TOL)
