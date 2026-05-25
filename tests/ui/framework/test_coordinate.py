"""
Comprehensive tests for the coordinate system module.

Tests cover:
- Point operations and properties
- Size operations and validation
- Rect operations and geometry
- Margins and insets
- Transform2D matrix operations
- Coordinate space conversions
- Anchor point calculations
"""

import math
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.ui.framework.coordinate import (
    CoordinateSpace,
    Anchor,
    StretchMode,
    Point,
    Size,
    Rect,
    Margins,
    Transform2D,
    CoordinateConverter,
    calculate_anchor_position,
)


class TestCoordinateSpace:
    """Tests for CoordinateSpace enum."""

    def test_pixel_space_is_valid_enum_member(self):
        """CoordinateSpace.PIXEL should be a valid enum member with auto() value."""
        assert isinstance(CoordinateSpace.PIXEL, CoordinateSpace)
        assert CoordinateSpace.PIXEL.name == "PIXEL"
        # auto() generates integer values, verify it has a value
        assert isinstance(CoordinateSpace.PIXEL.value, int)

    def test_normalized_space_is_valid_enum_member(self):
        """CoordinateSpace.NORMALIZED should be a valid enum member with auto() value."""
        assert isinstance(CoordinateSpace.NORMALIZED, CoordinateSpace)
        assert CoordinateSpace.NORMALIZED.name == "NORMALIZED"
        assert isinstance(CoordinateSpace.NORMALIZED.value, int)

    def test_viewport_space_is_valid_enum_member(self):
        """CoordinateSpace.VIEWPORT should be a valid enum member with auto() value."""
        assert isinstance(CoordinateSpace.VIEWPORT, CoordinateSpace)
        assert CoordinateSpace.VIEWPORT.name == "VIEWPORT"
        assert isinstance(CoordinateSpace.VIEWPORT.value, int)

    def test_parent_space_is_valid_enum_member(self):
        """CoordinateSpace.PARENT should be a valid enum member with auto() value."""
        assert isinstance(CoordinateSpace.PARENT, CoordinateSpace)
        assert CoordinateSpace.PARENT.name == "PARENT"
        assert isinstance(CoordinateSpace.PARENT.value, int)

    def test_spaces_are_distinct(self):
        """All coordinate spaces should have unique values."""
        spaces = [
            CoordinateSpace.PIXEL,
            CoordinateSpace.NORMALIZED,
            CoordinateSpace.VIEWPORT,
            CoordinateSpace.PARENT,
        ]
        values = [s.value for s in spaces]
        assert len(spaces) == len(set(spaces))
        assert len(values) == len(set(values))  # Values must also be unique

    def test_all_spaces_are_members_of_enum(self):
        """All coordinate spaces should be CoordinateSpace members."""
        for space in [CoordinateSpace.PIXEL, CoordinateSpace.NORMALIZED,
                      CoordinateSpace.VIEWPORT, CoordinateSpace.PARENT]:
            assert isinstance(space, CoordinateSpace)

    def test_enum_has_exactly_four_members(self):
        """CoordinateSpace should have exactly 4 members."""
        assert len(CoordinateSpace) == 4


class TestAnchor:
    """Tests for Anchor enum."""

    def test_top_left_anchor(self):
        """TOP_LEFT anchor should be (0, 0)."""
        assert Anchor.TOP_LEFT.x == 0.0
        assert Anchor.TOP_LEFT.y == 0.0

    def test_top_center_anchor(self):
        """TOP_CENTER anchor should be (0.5, 0)."""
        assert Anchor.TOP_CENTER.x == 0.5
        assert Anchor.TOP_CENTER.y == 0.0

    def test_top_right_anchor(self):
        """TOP_RIGHT anchor should be (1, 0)."""
        assert Anchor.TOP_RIGHT.x == 1.0
        assert Anchor.TOP_RIGHT.y == 0.0

    def test_center_left_anchor(self):
        """CENTER_LEFT anchor should be (0, 0.5)."""
        assert Anchor.CENTER_LEFT.x == 0.0
        assert Anchor.CENTER_LEFT.y == 0.5

    def test_center_anchor(self):
        """CENTER anchor should be (0.5, 0.5)."""
        assert Anchor.CENTER.x == 0.5
        assert Anchor.CENTER.y == 0.5

    def test_center_right_anchor(self):
        """CENTER_RIGHT anchor should be (1, 0.5)."""
        assert Anchor.CENTER_RIGHT.x == 1.0
        assert Anchor.CENTER_RIGHT.y == 0.5

    def test_bottom_left_anchor(self):
        """BOTTOM_LEFT anchor should be (0, 1)."""
        assert Anchor.BOTTOM_LEFT.x == 0.0
        assert Anchor.BOTTOM_LEFT.y == 1.0

    def test_bottom_center_anchor(self):
        """BOTTOM_CENTER anchor should be (0.5, 1)."""
        assert Anchor.BOTTOM_CENTER.x == 0.5
        assert Anchor.BOTTOM_CENTER.y == 1.0

    def test_bottom_right_anchor(self):
        """BOTTOM_RIGHT anchor should be (1, 1)."""
        assert Anchor.BOTTOM_RIGHT.x == 1.0
        assert Anchor.BOTTOM_RIGHT.y == 1.0

    def test_anchor_value_is_tuple(self):
        """Anchor value should be a tuple."""
        assert isinstance(Anchor.CENTER.value, tuple)
        assert len(Anchor.CENTER.value) == 2


class TestStretchMode:
    """Tests for StretchMode enum."""

    def test_none_mode_is_valid_enum_member(self):
        """StretchMode.NONE should be a valid enum member with auto() value."""
        assert isinstance(StretchMode.NONE, StretchMode)
        assert StretchMode.NONE.name == "NONE"
        assert isinstance(StretchMode.NONE.value, int)

    def test_horizontal_mode_is_valid_enum_member(self):
        """StretchMode.HORIZONTAL should be a valid enum member with auto() value."""
        assert isinstance(StretchMode.HORIZONTAL, StretchMode)
        assert StretchMode.HORIZONTAL.name == "HORIZONTAL"
        assert isinstance(StretchMode.HORIZONTAL.value, int)

    def test_vertical_mode_is_valid_enum_member(self):
        """StretchMode.VERTICAL should be a valid enum member with auto() value."""
        assert isinstance(StretchMode.VERTICAL, StretchMode)
        assert StretchMode.VERTICAL.name == "VERTICAL"
        assert isinstance(StretchMode.VERTICAL.value, int)

    def test_both_mode_is_valid_enum_member(self):
        """StretchMode.BOTH should be a valid enum member with auto() value."""
        assert isinstance(StretchMode.BOTH, StretchMode)
        assert StretchMode.BOTH.name == "BOTH"
        assert isinstance(StretchMode.BOTH.value, int)

    def test_stretch_modes_are_distinct(self):
        """All stretch modes should have unique values."""
        modes = [
            StretchMode.NONE,
            StretchMode.HORIZONTAL,
            StretchMode.VERTICAL,
            StretchMode.BOTH,
        ]
        values = [m.value for m in modes]
        assert len(modes) == len(set(modes))
        assert len(values) == len(set(values))  # Values must also be unique

    def test_all_modes_are_members_of_enum(self):
        """All stretch modes should be StretchMode members."""
        for mode in [StretchMode.NONE, StretchMode.HORIZONTAL,
                     StretchMode.VERTICAL, StretchMode.BOTH]:
            assert isinstance(mode, StretchMode)

    def test_enum_has_exactly_four_members(self):
        """StretchMode should have exactly 4 members."""
        assert len(StretchMode) == 4


class TestPoint:
    """Tests for Point dataclass."""

    def test_default_point_is_origin(self):
        """Default Point should be at origin."""
        p = Point()
        assert p.x == 0.0
        assert p.y == 0.0

    def test_point_with_values(self):
        """Point should store x and y values."""
        p = Point(10.0, 20.0)
        assert p.x == 10.0
        assert p.y == 20.0

    def test_point_addition(self):
        """Point addition should work correctly."""
        p1 = Point(1.0, 2.0)
        p2 = Point(3.0, 4.0)
        result = p1 + p2
        assert result.x == 4.0
        assert result.y == 6.0

    def test_point_subtraction(self):
        """Point subtraction should work correctly."""
        p1 = Point(5.0, 10.0)
        p2 = Point(2.0, 3.0)
        result = p1 - p2
        assert result.x == 3.0
        assert result.y == 7.0

    def test_point_multiplication(self):
        """Point scalar multiplication should work correctly."""
        p = Point(3.0, 4.0)
        result = p * 2.0
        assert result.x == 6.0
        assert result.y == 8.0

    def test_point_division(self):
        """Point scalar division should work correctly."""
        p = Point(10.0, 20.0)
        result = p / 2.0
        assert result.x == 5.0
        assert result.y == 10.0

    def test_point_division_by_zero_raises(self):
        """Point division by zero should raise ZeroDivisionError."""
        p = Point(10.0, 20.0)
        with pytest.raises(ZeroDivisionError):
            _ = p / 0.0

    def test_point_negation(self):
        """Point negation should work correctly."""
        p = Point(3.0, -4.0)
        result = -p
        assert result.x == -3.0
        assert result.y == 4.0

    def test_point_equality(self):
        """Point equality should compare values."""
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0, 2.0)
        assert p1 == p2

    def test_point_equality_with_tolerance(self):
        """Point equality should handle floating point tolerance."""
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0 + 1e-10, 2.0 - 1e-10)
        assert p1 == p2

    def test_point_inequality(self):
        """Point inequality should work correctly."""
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0, 3.0)
        assert p1 != p2

    def test_point_hash(self):
        """Point should be hashable and equal points should have equal hashes."""
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0, 2.0)
        p3 = Point(3.0, 4.0)
        # Same values should produce same hash
        assert hash(p1) == hash(p2)
        # Different values should (very likely) produce different hashes
        assert hash(p1) != hash(p3)
        # Point can be used in sets/dicts
        point_set = {p1, p2, p3}
        assert len(point_set) == 2  # p1 and p2 are equal

    def test_point_distance_to(self):
        """distance_to should calculate Euclidean distance."""
        p1 = Point(0.0, 0.0)
        p2 = Point(3.0, 4.0)
        assert p1.distance_to(p2) == 5.0

    def test_point_distance_to_self(self):
        """Distance to self should be zero."""
        p = Point(5.0, 5.0)
        assert p.distance_to(p) == 0.0

    def test_point_lerp(self):
        """lerp should interpolate between points."""
        p1 = Point(0.0, 0.0)
        p2 = Point(10.0, 10.0)
        result = p1.lerp(p2, 0.5)
        assert result.x == 5.0
        assert result.y == 5.0

    def test_point_lerp_at_zero(self):
        """lerp at t=0 should return first point."""
        p1 = Point(1.0, 2.0)
        p2 = Point(10.0, 20.0)
        result = p1.lerp(p2, 0.0)
        assert result == p1

    def test_point_lerp_at_one(self):
        """lerp at t=1 should return second point."""
        p1 = Point(1.0, 2.0)
        p2 = Point(10.0, 20.0)
        result = p1.lerp(p2, 1.0)
        assert result == p2

    def test_point_as_tuple(self):
        """as_tuple should return (x, y)."""
        p = Point(3.0, 4.0)
        assert p.as_tuple() == (3.0, 4.0)

    def test_point_from_tuple(self):
        """from_tuple should create Point from tuple."""
        p = Point.from_tuple((5.0, 6.0))
        assert p.x == 5.0
        assert p.y == 6.0

    def test_point_zero(self):
        """zero() should return origin point."""
        p = Point.zero()
        assert p.x == 0.0
        assert p.y == 0.0

    def test_point_one(self):
        """one() should return (1, 1) point."""
        p = Point.one()
        assert p.x == 1.0
        assert p.y == 1.0


class TestSize:
    """Tests for Size dataclass."""

    def test_default_size_is_zero(self):
        """Default Size should be zero."""
        s = Size()
        assert s.width == 0.0
        assert s.height == 0.0

    def test_size_with_values(self):
        """Size should store width and height values."""
        s = Size(100.0, 200.0)
        assert s.width == 100.0
        assert s.height == 200.0

    def test_negative_width_raises(self):
        """Negative width should raise ValueError."""
        with pytest.raises(ValueError, match="Width cannot be negative"):
            Size(-1.0, 10.0)

    def test_negative_height_raises(self):
        """Negative height should raise ValueError."""
        with pytest.raises(ValueError, match="Height cannot be negative"):
            Size(10.0, -1.0)

    def test_size_addition(self):
        """Size addition should work correctly."""
        s1 = Size(10.0, 20.0)
        s2 = Size(5.0, 15.0)
        result = s1 + s2
        assert result.width == 15.0
        assert result.height == 35.0

    def test_size_multiplication(self):
        """Size scalar multiplication should work correctly."""
        s = Size(10.0, 20.0)
        result = s * 2.0
        assert result.width == 20.0
        assert result.height == 40.0

    def test_size_equality(self):
        """Size equality should compare values."""
        s1 = Size(100.0, 200.0)
        s2 = Size(100.0, 200.0)
        assert s1 == s2

    def test_size_hash(self):
        """Size should be hashable and equal sizes should have equal hashes."""
        s1 = Size(100.0, 200.0)
        s2 = Size(100.0, 200.0)
        s3 = Size(50.0, 75.0)
        # Same values should produce same hash
        assert hash(s1) == hash(s2)
        # Different values should (very likely) produce different hashes
        assert hash(s1) != hash(s3)
        # Size can be used in sets/dicts
        size_set = {s1, s2, s3}
        assert len(size_set) == 2  # s1 and s2 are equal

    def test_size_area(self):
        """area should calculate width * height."""
        s = Size(10.0, 20.0)
        assert s.area == 200.0

    def test_size_area_zero(self):
        """area should be zero if any dimension is zero."""
        s = Size(10.0, 0.0)
        assert s.area == 0.0

    def test_size_aspect_ratio(self):
        """aspect_ratio should calculate width/height."""
        s = Size(16.0, 9.0)
        assert math.isclose(s.aspect_ratio, 16.0 / 9.0)

    def test_size_aspect_ratio_zero_height(self):
        """aspect_ratio should return 0 for zero height."""
        s = Size(10.0, 0.0)
        assert s.aspect_ratio == 0.0

    def test_size_contains_point_inside(self):
        """contains should return True for point inside."""
        s = Size(100.0, 100.0)
        assert s.contains(Point(50.0, 50.0))

    def test_size_contains_point_on_boundary(self):
        """contains should return True for point on boundary."""
        s = Size(100.0, 100.0)
        assert s.contains(Point(0.0, 0.0))
        assert s.contains(Point(100.0, 100.0))

    def test_size_contains_point_outside(self):
        """contains should return False for point outside."""
        s = Size(100.0, 100.0)
        assert not s.contains(Point(150.0, 50.0))
        assert not s.contains(Point(-10.0, 50.0))

    def test_size_as_tuple(self):
        """as_tuple should return (width, height)."""
        s = Size(100.0, 200.0)
        assert s.as_tuple() == (100.0, 200.0)

    def test_size_as_point(self):
        """as_point should convert to Point."""
        s = Size(100.0, 200.0)
        p = s.as_point()
        assert p.x == 100.0
        assert p.y == 200.0

    def test_size_from_tuple(self):
        """from_tuple should create Size from tuple."""
        s = Size.from_tuple((100.0, 200.0))
        assert s.width == 100.0
        assert s.height == 200.0

    def test_size_zero(self):
        """zero() should return zero size."""
        s = Size.zero()
        assert s.width == 0.0
        assert s.height == 0.0

    def test_size_square(self):
        """square() should create square size."""
        s = Size.square(50.0)
        assert s.width == 50.0
        assert s.height == 50.0


class TestRect:
    """Tests for Rect dataclass."""

    def test_default_rect_is_zero(self):
        """Default Rect should be at origin with zero size."""
        r = Rect()
        assert r.x == 0.0
        assert r.y == 0.0
        assert r.width == 0.0
        assert r.height == 0.0

    def test_rect_with_values(self):
        """Rect should store position and size."""
        r = Rect(10.0, 20.0, 100.0, 200.0)
        assert r.x == 10.0
        assert r.y == 20.0
        assert r.width == 100.0
        assert r.height == 200.0

    def test_negative_width_raises(self):
        """Negative width should raise ValueError."""
        with pytest.raises(ValueError, match="Width cannot be negative"):
            Rect(0.0, 0.0, -10.0, 10.0)

    def test_negative_height_raises(self):
        """Negative height should raise ValueError."""
        with pytest.raises(ValueError, match="Height cannot be negative"):
            Rect(0.0, 0.0, 10.0, -10.0)

    def test_rect_position_property(self):
        """position property should return Point."""
        r = Rect(10.0, 20.0, 100.0, 200.0)
        assert r.position == Point(10.0, 20.0)

    def test_rect_position_setter(self):
        """position setter should update x and y."""
        r = Rect(10.0, 20.0, 100.0, 200.0)
        r.position = Point(30.0, 40.0)
        assert r.x == 30.0
        assert r.y == 40.0

    def test_rect_size_property(self):
        """size property should return Size."""
        r = Rect(10.0, 20.0, 100.0, 200.0)
        assert r.size == Size(100.0, 200.0)

    def test_rect_size_setter(self):
        """size setter should update width and height."""
        r = Rect(10.0, 20.0, 100.0, 200.0)
        r.size = Size(50.0, 60.0)
        assert r.width == 50.0
        assert r.height == 60.0

    def test_rect_edge_properties(self):
        """Edge properties should return correct values."""
        r = Rect(10.0, 20.0, 100.0, 200.0)
        assert r.left == 10.0
        assert r.top == 20.0
        assert r.right == 110.0
        assert r.bottom == 220.0

    def test_rect_center(self):
        """center should return center point."""
        r = Rect(0.0, 0.0, 100.0, 200.0)
        assert r.center == Point(50.0, 100.0)

    def test_rect_corners(self):
        """Corner properties should return correct points."""
        r = Rect(10.0, 20.0, 100.0, 200.0)
        assert r.top_left == Point(10.0, 20.0)
        assert r.top_right == Point(110.0, 20.0)
        assert r.bottom_left == Point(10.0, 220.0)
        assert r.bottom_right == Point(110.0, 220.0)

    def test_rect_area(self):
        """area should calculate width * height."""
        r = Rect(0.0, 0.0, 10.0, 20.0)
        assert r.area == 200.0

    def test_rect_contains_point_inside(self):
        """contains_point should return True for point inside."""
        r = Rect(0.0, 0.0, 100.0, 100.0)
        assert r.contains_point(Point(50.0, 50.0))

    def test_rect_contains_point_on_boundary(self):
        """contains_point should return True for point on boundary."""
        r = Rect(0.0, 0.0, 100.0, 100.0)
        assert r.contains_point(Point(0.0, 0.0))
        assert r.contains_point(Point(100.0, 100.0))

    def test_rect_contains_point_outside(self):
        """contains_point should return False for point outside."""
        r = Rect(0.0, 0.0, 100.0, 100.0)
        assert not r.contains_point(Point(150.0, 50.0))
        assert not r.contains_point(Point(-10.0, 50.0))

    def test_rect_contains_rect_fully_inside(self):
        """contains_rect should return True when fully contained."""
        outer = Rect(0.0, 0.0, 100.0, 100.0)
        inner = Rect(10.0, 10.0, 50.0, 50.0)
        assert outer.contains_rect(inner)

    def test_rect_contains_rect_same_size(self):
        """contains_rect should return True for same rect."""
        r = Rect(0.0, 0.0, 100.0, 100.0)
        assert r.contains_rect(r)

    def test_rect_contains_rect_partial_overlap(self):
        """contains_rect should return False for partial overlap."""
        r1 = Rect(0.0, 0.0, 100.0, 100.0)
        r2 = Rect(50.0, 50.0, 100.0, 100.0)
        assert not r1.contains_rect(r2)

    def test_rect_intersects_overlap(self):
        """intersects should return True for overlapping rects."""
        r1 = Rect(0.0, 0.0, 100.0, 100.0)
        r2 = Rect(50.0, 50.0, 100.0, 100.0)
        assert r1.intersects(r2)
        assert r2.intersects(r1)

    def test_rect_intersects_no_overlap(self):
        """intersects should return False for non-overlapping rects."""
        r1 = Rect(0.0, 0.0, 100.0, 100.0)
        r2 = Rect(200.0, 200.0, 100.0, 100.0)
        assert not r1.intersects(r2)

    def test_rect_intersects_touching(self):
        """intersects should return True for touching rects."""
        r1 = Rect(0.0, 0.0, 100.0, 100.0)
        r2 = Rect(100.0, 0.0, 100.0, 100.0)
        assert r1.intersects(r2)

    def test_rect_intersection_overlap(self):
        """intersection should return overlap rect."""
        r1 = Rect(0.0, 0.0, 100.0, 100.0)
        r2 = Rect(50.0, 50.0, 100.0, 100.0)
        result = r1.intersection(r2)
        assert result is not None
        assert result.x == 50.0
        assert result.y == 50.0
        assert result.width == 50.0
        assert result.height == 50.0

    def test_rect_intersection_no_overlap(self):
        """intersection should return None for non-overlapping."""
        r1 = Rect(0.0, 0.0, 100.0, 100.0)
        r2 = Rect(200.0, 200.0, 100.0, 100.0)
        assert r1.intersection(r2) is None

    def test_rect_union(self):
        """union should return bounding rect."""
        r1 = Rect(0.0, 0.0, 100.0, 100.0)
        r2 = Rect(50.0, 50.0, 100.0, 100.0)
        result = r1.union(r2)
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.width == 150.0
        assert result.height == 150.0

    def test_rect_expand(self):
        """expand should grow rect on all sides."""
        r = Rect(10.0, 10.0, 100.0, 100.0)
        result = r.expand(5.0)
        assert result.x == 5.0
        assert result.y == 5.0
        assert result.width == 110.0
        assert result.height == 110.0

    def test_rect_contract(self):
        """contract should shrink rect on all sides."""
        r = Rect(10.0, 10.0, 100.0, 100.0)
        result = r.contract(5.0)
        assert result.x == 15.0
        assert result.y == 15.0
        assert result.width == 90.0
        assert result.height == 90.0

    def test_rect_contract_to_zero(self):
        """contract should clamp to zero size."""
        r = Rect(0.0, 0.0, 10.0, 10.0)
        result = r.contract(20.0)
        assert result.width == 0.0
        assert result.height == 0.0

    def test_rect_translate(self):
        """translate should move rect by offset."""
        r = Rect(10.0, 10.0, 100.0, 100.0)
        result = r.translate(Point(5.0, -5.0))
        assert result.x == 15.0
        assert result.y == 5.0
        assert result.width == 100.0
        assert result.height == 100.0

    def test_rect_from_points(self):
        """from_points should create rect from corners."""
        r = Rect.from_points(Point(10.0, 20.0), Point(110.0, 220.0))
        assert r.x == 10.0
        assert r.y == 20.0
        assert r.width == 100.0
        assert r.height == 200.0

    def test_rect_from_points_reversed(self):
        """from_points should handle reversed corners."""
        r = Rect.from_points(Point(110.0, 220.0), Point(10.0, 20.0))
        assert r.x == 10.0
        assert r.y == 20.0
        assert r.width == 100.0
        assert r.height == 200.0

    def test_rect_from_center(self):
        """from_center should create centered rect."""
        r = Rect.from_center(Point(50.0, 50.0), Size(100.0, 100.0))
        assert r.x == 0.0
        assert r.y == 0.0
        assert r.width == 100.0
        assert r.height == 100.0

    def test_rect_zero(self):
        """zero() should return zero rect at origin."""
        r = Rect.zero()
        assert r.x == 0.0
        assert r.y == 0.0
        assert r.width == 0.0
        assert r.height == 0.0


class TestMargins:
    """Tests for Margins dataclass."""

    def test_default_margins_are_zero(self):
        """Default Margins should be zero on all sides."""
        m = Margins()
        assert m.top == 0.0
        assert m.right == 0.0
        assert m.bottom == 0.0
        assert m.left == 0.0

    def test_margins_with_values(self):
        """Margins should store values for all sides."""
        m = Margins(10.0, 20.0, 30.0, 40.0)
        assert m.top == 10.0
        assert m.right == 20.0
        assert m.bottom == 30.0
        assert m.left == 40.0

    def test_negative_margin_raises(self):
        """Negative margins should raise ValueError."""
        with pytest.raises(ValueError, match="Margin .* cannot be negative"):
            Margins(-1.0, 0.0, 0.0, 0.0)

    def test_margins_horizontal(self):
        """horizontal should return left + right."""
        m = Margins(10.0, 20.0, 30.0, 40.0)
        assert m.horizontal == 60.0

    def test_margins_vertical(self):
        """vertical should return top + bottom."""
        m = Margins(10.0, 20.0, 30.0, 40.0)
        assert m.vertical == 40.0

    def test_margins_apply_to_rect(self):
        """apply_to_rect should shrink rect by margins."""
        r = Rect(0.0, 0.0, 100.0, 100.0)
        m = Margins(10.0, 10.0, 10.0, 10.0)
        result = m.apply_to_rect(r)
        assert result.x == 10.0
        assert result.y == 10.0
        assert result.width == 80.0
        assert result.height == 80.0

    def test_margins_apply_to_rect_clamps_size(self):
        """apply_to_rect should clamp size to zero."""
        r = Rect(0.0, 0.0, 10.0, 10.0)
        m = Margins(20.0, 20.0, 20.0, 20.0)
        result = m.apply_to_rect(r)
        assert result.width == 0.0
        assert result.height == 0.0

    def test_margins_all(self):
        """all() should create uniform margins."""
        m = Margins.all(15.0)
        assert m.top == 15.0
        assert m.right == 15.0
        assert m.bottom == 15.0
        assert m.left == 15.0

    def test_margins_symmetric(self):
        """symmetric() should create symmetric margins."""
        m = Margins.symmetric(10.0, 20.0)
        assert m.left == 10.0
        assert m.right == 10.0
        assert m.top == 20.0
        assert m.bottom == 20.0

    def test_margins_zero(self):
        """zero() should create zero margins."""
        m = Margins.zero()
        assert m.top == 0.0
        assert m.right == 0.0
        assert m.bottom == 0.0
        assert m.left == 0.0


class TestTransform2D:
    """Tests for Transform2D dataclass."""

    def test_default_transform_is_identity(self):
        """Default Transform2D should be identity."""
        t = Transform2D()
        assert t.position == Point.zero()
        assert t.rotation == 0.0
        assert t.scale == Point.one()

    def test_transform_with_position(self):
        """Transform should store position."""
        t = Transform2D(position=Point(10.0, 20.0))
        assert t.position == Point(10.0, 20.0)

    def test_transform_with_rotation(self):
        """Transform should store rotation in radians."""
        t = Transform2D(rotation=math.pi / 2)
        assert t.rotation == math.pi / 2

    def test_transform_rotation_degrees(self):
        """rotation_degrees property should convert radians to degrees."""
        t = Transform2D(rotation=math.pi)
        assert math.isclose(t.rotation_degrees, 180.0)

    def test_transform_rotation_degrees_setter(self):
        """rotation_degrees setter should convert degrees to radians."""
        t = Transform2D()
        t.rotation_degrees = 90.0
        assert math.isclose(t.rotation, math.pi / 2)

    def test_transform_with_scale(self):
        """Transform should store scale."""
        t = Transform2D(scale=Point(2.0, 2.0))
        assert t.scale == Point(2.0, 2.0)

    def test_transform_point_translation(self):
        """transform_point should apply translation."""
        t = Transform2D(position=Point(10.0, 20.0))
        p = Point(5.0, 5.0)
        result = t.transform_point(p)
        assert result == Point(15.0, 25.0)

    def test_transform_point_scale(self):
        """transform_point should apply scale."""
        t = Transform2D(scale=Point(2.0, 3.0))
        p = Point(5.0, 5.0)
        result = t.transform_point(p)
        assert result == Point(10.0, 15.0)

    def test_transform_point_rotation_90(self):
        """transform_point should apply 90 degree rotation."""
        t = Transform2D(rotation=math.pi / 2)
        p = Point(1.0, 0.0)
        result = t.transform_point(p)
        assert math.isclose(result.x, 0.0, abs_tol=1e-9)
        assert math.isclose(result.y, 1.0)

    def test_transform_point_combined(self):
        """transform_point should apply scale, rotate, translate in order."""
        t = Transform2D(
            position=Point(100.0, 100.0),
            scale=Point(2.0, 2.0),
        )
        p = Point(5.0, 5.0)
        result = t.transform_point(p)
        assert result == Point(110.0, 110.0)

    def test_inverse_transform_point(self):
        """inverse_transform_point should reverse transformation."""
        t = Transform2D(
            position=Point(10.0, 20.0),
            scale=Point(2.0, 2.0),
        )
        p = Point(30.0, 60.0)
        result = t.inverse_transform_point(p)
        assert result == Point(10.0, 20.0)

    def test_transform_inverse_round_trip(self):
        """transform then inverse should return original point."""
        t = Transform2D(
            position=Point(50.0, 50.0),
            rotation=math.pi / 4,
            scale=Point(2.0, 2.0),
        )
        original = Point(10.0, 20.0)
        transformed = t.transform_point(original)
        recovered = t.inverse_transform_point(transformed)
        assert math.isclose(recovered.x, original.x, abs_tol=1e-9)
        assert math.isclose(recovered.y, original.y, abs_tol=1e-9)

    def test_transform_compose(self):
        """compose should combine transforms."""
        t1 = Transform2D(position=Point(10.0, 0.0))
        t2 = Transform2D(position=Point(0.0, 20.0))
        combined = t1.compose(t2)
        p = Point(0.0, 0.0)
        result = combined.transform_point(p)
        assert result.x == 10.0
        assert result.y == 20.0

    def test_transform_identity(self):
        """identity() should return identity transform."""
        t = Transform2D.identity()
        assert t.position == Point.zero()
        assert t.rotation == 0.0
        assert t.scale == Point.one()

    def test_transform_from_translation(self):
        """from_translation should create translation-only transform."""
        t = Transform2D.from_translation(Point(10.0, 20.0))
        assert t.position == Point(10.0, 20.0)
        assert t.rotation == 0.0
        assert t.scale == Point.one()

    def test_transform_from_rotation(self):
        """from_rotation should create rotation-only transform."""
        t = Transform2D.from_rotation(math.pi)
        assert t.position == Point.zero()
        assert t.rotation == math.pi
        assert t.scale == Point.one()

    def test_transform_from_scale_uniform(self):
        """from_scale with scalar should create uniform scale."""
        t = Transform2D.from_scale(2.0)
        assert t.position == Point.zero()
        assert t.rotation == 0.0
        assert t.scale == Point(2.0, 2.0)

    def test_transform_from_scale_non_uniform(self):
        """from_scale with Point should create non-uniform scale."""
        t = Transform2D.from_scale(Point(2.0, 3.0))
        assert t.scale == Point(2.0, 3.0)


class TestCoordinateConverter:
    """Tests for CoordinateConverter class."""

    def test_default_viewport_size(self):
        """Default viewport size should be 1920x1080."""
        conv = CoordinateConverter()
        assert conv.viewport_size == Size(1920, 1080)

    def test_default_dpi_scale(self):
        """Default DPI scale should be 1.0."""
        conv = CoordinateConverter()
        assert conv.dpi_scale == 1.0

    def test_viewport_size_setter(self):
        """viewport_size should be settable."""
        conv = CoordinateConverter()
        conv.viewport_size = Size(1280, 720)
        assert conv.viewport_size == Size(1280, 720)

    def test_dpi_scale_setter(self):
        """dpi_scale should be settable."""
        conv = CoordinateConverter()
        conv.dpi_scale = 2.0
        assert conv.dpi_scale == 2.0

    def test_dpi_scale_invalid_raises(self):
        """Invalid dpi_scale should raise ValueError."""
        conv = CoordinateConverter()
        with pytest.raises(ValueError, match="DPI scale must be positive"):
            conv.dpi_scale = 0.0

    def test_to_pixels_from_pixel(self):
        """to_pixels from PIXEL should return unchanged."""
        conv = CoordinateConverter()
        p = Point(100.0, 200.0)
        result = conv.to_pixels(p, CoordinateSpace.PIXEL)
        assert result == p

    def test_to_pixels_from_viewport(self):
        """to_pixels from VIEWPORT should scale by viewport size."""
        conv = CoordinateConverter(viewport_size=Size(1000, 500))
        p = Point(0.5, 0.5)
        result = conv.to_pixels(p, CoordinateSpace.VIEWPORT)
        assert result == Point(500.0, 250.0)

    def test_to_pixels_from_normalized_requires_parent(self):
        """to_pixels from NORMALIZED should require parent_rect."""
        conv = CoordinateConverter()
        with pytest.raises(ValueError, match="parent_rect required"):
            conv.to_pixels(Point(0.5, 0.5), CoordinateSpace.NORMALIZED)

    def test_to_pixels_from_normalized(self):
        """to_pixels from NORMALIZED should use parent_rect."""
        conv = CoordinateConverter()
        parent = Rect(100.0, 100.0, 200.0, 200.0)
        p = Point(0.5, 0.5)
        result = conv.to_pixels(p, CoordinateSpace.NORMALIZED, parent)
        assert result == Point(200.0, 200.0)

    def test_from_pixels_to_pixel(self):
        """from_pixels to PIXEL should return unchanged."""
        conv = CoordinateConverter()
        p = Point(100.0, 200.0)
        result = conv.from_pixels(p, CoordinateSpace.PIXEL)
        assert result == p

    def test_from_pixels_to_viewport(self):
        """from_pixels to VIEWPORT should normalize by viewport size."""
        conv = CoordinateConverter(viewport_size=Size(1000, 500))
        p = Point(500.0, 250.0)
        result = conv.from_pixels(p, CoordinateSpace.VIEWPORT)
        assert result == Point(0.5, 0.5)

    def test_from_pixels_to_viewport_zero_size(self):
        """from_pixels to VIEWPORT with zero size should return zero."""
        conv = CoordinateConverter(viewport_size=Size(0, 0))
        p = Point(100.0, 100.0)
        result = conv.from_pixels(p, CoordinateSpace.VIEWPORT)
        assert result == Point.zero()

    def test_from_pixels_to_normalized(self):
        """from_pixels to NORMALIZED should use parent_rect."""
        conv = CoordinateConverter()
        parent = Rect(100.0, 100.0, 200.0, 200.0)
        p = Point(200.0, 200.0)
        result = conv.from_pixels(p, CoordinateSpace.NORMALIZED, parent)
        assert result == Point(0.5, 0.5)

    def test_convert_same_space(self):
        """convert with same space should return unchanged."""
        conv = CoordinateConverter()
        p = Point(100.0, 200.0)
        result = conv.convert(p, CoordinateSpace.PIXEL, CoordinateSpace.PIXEL)
        assert result == p

    def test_convert_viewport_to_pixel(self):
        """convert from VIEWPORT to PIXEL should scale."""
        conv = CoordinateConverter(viewport_size=Size(1000, 500))
        p = Point(0.5, 0.5)
        result = conv.convert(
            p, CoordinateSpace.VIEWPORT, CoordinateSpace.PIXEL
        )
        assert result == Point(500.0, 250.0)

    def test_convert_pixel_to_viewport(self):
        """convert from PIXEL to VIEWPORT should normalize."""
        conv = CoordinateConverter(viewport_size=Size(1000, 500))
        p = Point(500.0, 250.0)
        result = conv.convert(
            p, CoordinateSpace.PIXEL, CoordinateSpace.VIEWPORT
        )
        assert result == Point(0.5, 0.5)

    def test_apply_dpi_scale(self):
        """apply_dpi_scale should multiply by scale factor."""
        conv = CoordinateConverter(dpi_scale=2.0)
        assert conv.apply_dpi_scale(100.0) == 200.0

    def test_remove_dpi_scale(self):
        """remove_dpi_scale should divide by scale factor."""
        conv = CoordinateConverter(dpi_scale=2.0)
        assert conv.remove_dpi_scale(200.0) == 100.0


class TestCalculateAnchorPosition:
    """Tests for calculate_anchor_position function."""

    def test_top_left_anchor(self):
        """TOP_LEFT anchor should position at origin."""
        pos = calculate_anchor_position(
            Anchor.TOP_LEFT,
            Size(1000, 1000),
            Size(100, 100),
        )
        assert pos == Point(0.0, 0.0)

    def test_center_anchor(self):
        """CENTER anchor should center widget."""
        pos = calculate_anchor_position(
            Anchor.CENTER,
            Size(1000, 1000),
            Size(100, 100),
        )
        assert pos == Point(450.0, 450.0)

    def test_bottom_right_anchor(self):
        """BOTTOM_RIGHT anchor should position at bottom-right."""
        pos = calculate_anchor_position(
            Anchor.BOTTOM_RIGHT,
            Size(1000, 1000),
            Size(100, 100),
        )
        assert pos == Point(900.0, 900.0)

    def test_anchor_with_custom_pivot(self):
        """Custom pivot should offset from anchor."""
        pos = calculate_anchor_position(
            Anchor.TOP_LEFT,
            Size(1000, 1000),
            Size(100, 100),
            pivot=Point(0.5, 0.5),  # Center pivot
        )
        assert pos == Point(-50.0, -50.0)

    def test_anchor_with_margins_left(self):
        """Margins should offset from left anchor."""
        pos = calculate_anchor_position(
            Anchor.TOP_LEFT,
            Size(1000, 1000),
            Size(100, 100),
            margins=Margins(10.0, 0.0, 0.0, 20.0),
        )
        assert pos == Point(20.0, 10.0)

    def test_anchor_with_margins_right(self):
        """Margins should offset from right anchor."""
        pos = calculate_anchor_position(
            Anchor.TOP_RIGHT,
            Size(1000, 1000),
            Size(100, 100),
            margins=Margins(10.0, 20.0, 0.0, 0.0),
        )
        assert pos == Point(880.0, 10.0)

    def test_anchor_with_margins_bottom(self):
        """Margins should offset from bottom anchor."""
        pos = calculate_anchor_position(
            Anchor.BOTTOM_LEFT,
            Size(1000, 1000),
            Size(100, 100),
            margins=Margins(0.0, 0.0, 20.0, 10.0),
        )
        assert pos == Point(10.0, 880.0)

    def test_center_anchor_ignores_margins(self):
        """CENTER anchor should not be affected by margins."""
        pos_no_margins = calculate_anchor_position(
            Anchor.CENTER,
            Size(1000, 1000),
            Size(100, 100),
        )
        pos_with_margins = calculate_anchor_position(
            Anchor.CENTER,
            Size(1000, 1000),
            Size(100, 100),
            margins=Margins(10.0, 10.0, 10.0, 10.0),
        )
        assert pos_no_margins == pos_with_margins
