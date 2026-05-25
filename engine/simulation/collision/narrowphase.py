"""
Narrowphase Collision Detection Algorithms.

This module implements precise collision detection algorithms:
- GJK (Gilbert-Johnson-Keerthi) - distance and intersection detection
- EPA (Expanding Polytope Algorithm) - penetration depth and normal
- SAT (Separating Axis Theorem) - fast box-box collision
- Specialized shape-pair tests for spheres, capsules, and boxes
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
import math

from .broadphase import Vec3, AABB
from .config import (
    CONTACT_TOLERANCE,
    GJK_MAX_ITERATIONS,
    EPA_MAX_ITERATIONS,
    EPA_TOLERANCE,
    SAT_EDGE_BIAS,
    MAX_CONTACT_POINTS,
    NUMERICAL_EPSILON,
    PARALLEL_THRESHOLD,
)


# =============================================================================
# Enums and Data Structures
# =============================================================================


class NarrowphaseAlgorithm(Enum):
    """Available narrowphase algorithms."""

    GJK_EPA = auto()  # General convex shapes
    SAT = auto()       # Fast box-box
    MPR = auto()       # Minkowski Portal Refinement


class ShapeType(Enum):
    """Supported collision shape types."""

    SPHERE = auto()
    CAPSULE = auto()
    BOX = auto()
    CONVEX_HULL = auto()
    MESH = auto()


@dataclass
class ContactResult:
    """Result of a narrowphase collision test."""

    colliding: bool = False
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    depth: float = 0.0
    points: list[Vec3] = field(default_factory=list)
    distance: float = float("inf")

    def __bool__(self) -> bool:
        return self.colliding


@dataclass
class Sphere:
    """Sphere collision shape."""

    center: Vec3 = field(default_factory=Vec3)
    radius: float = 1.0

    def support(self, direction: Vec3) -> Vec3:
        """Get support point in given direction."""
        d = direction.normalized()
        return self.center + d * self.radius


@dataclass
class Capsule:
    """Capsule collision shape (line segment with radius)."""

    start: Vec3 = field(default_factory=Vec3)
    end: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    radius: float = 0.5

    @property
    def axis(self) -> Vec3:
        """Get capsule axis vector."""
        return self.end - self.start

    @property
    def height(self) -> float:
        """Get capsule height (end-to-end)."""
        return self.axis.length()

    def support(self, direction: Vec3) -> Vec3:
        """Get support point in given direction."""
        # Project direction onto axis
        axis = self.axis
        axis_length = axis.length()
        if axis_length < 1e-10:
            return self.start + direction.normalized() * self.radius

        axis_dir = axis * (1.0 / axis_length)
        dot = direction.dot(axis_dir)

        # Choose endpoint furthest in direction
        if dot > 0:
            point = self.end
        else:
            point = self.start

        # Add sphere contribution
        d = direction.normalized()
        return point + d * self.radius

    def closest_point_on_axis(self, point: Vec3) -> Vec3:
        """Get closest point on capsule axis to given point."""
        ab = self.end - self.start
        ab_dot_ab = ab.dot(ab)
        # Handle degenerate capsule (point capsule)
        if ab_dot_ab < 1e-10:
            return Vec3(self.start.x, self.start.y, self.start.z)
        t = (point - self.start).dot(ab) / ab_dot_ab
        t = max(0.0, min(1.0, t))
        return self.start + ab * t


@dataclass
class Box:
    """Oriented bounding box collision shape."""

    center: Vec3 = field(default_factory=Vec3)
    half_extents: Vec3 = field(default_factory=lambda: Vec3(0.5, 0.5, 0.5))
    # Rotation as 3x3 matrix (row-major: axes[0] is x-axis, etc.)
    axes: tuple[Vec3, Vec3, Vec3] = field(
        default_factory=lambda: (Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))
    )

    def support(self, direction: Vec3) -> Vec3:
        """Get support point in given direction."""
        result = Vec3(self.center.x, self.center.y, self.center.z)
        for i, axis in enumerate(self.axes):
            extent = [self.half_extents.x, self.half_extents.y, self.half_extents.z][i]
            sign = 1.0 if direction.dot(axis) > 0 else -1.0
            result = result + axis * (extent * sign)
        return result

    def get_vertices(self) -> list[Vec3]:
        """Get all 8 vertices of the box."""
        vertices: list[Vec3] = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    v = self.center
                    v = v + self.axes[0] * (self.half_extents.x * sx)
                    v = v + self.axes[1] * (self.half_extents.y * sy)
                    v = v + self.axes[2] * (self.half_extents.z * sz)
                    vertices.append(v)
        return vertices

    @classmethod
    def from_aabb(cls, aabb: AABB) -> "Box":
        """Create axis-aligned box from AABB."""
        center = aabb.center()
        half_extents = aabb.extents()
        return cls(center, half_extents)


@dataclass
class ConvexHull:
    """Convex hull collision shape."""

    vertices: list[Vec3] = field(default_factory=list)
    _support_cache: dict[tuple[int, int, int], int] = field(
        default_factory=dict, repr=False
    )

    def support(self, direction: Vec3) -> Vec3:
        """Get support point in given direction using hill climbing."""
        if not self.vertices:
            # Return origin for empty hull - caller should validate hull before use
            raise ValueError("ConvexHull has no vertices - cannot compute support point")

        best_dot = float("-inf")
        best_vertex = self.vertices[0]

        for vertex in self.vertices:
            dot = direction.dot(vertex)
            if dot > best_dot:
                best_dot = dot
                best_vertex = vertex

        return best_vertex


# =============================================================================
# GJK Algorithm
# =============================================================================


@dataclass
class SimplexVertex:
    """Vertex in GJK simplex."""

    point: Vec3  # Point on Minkowski difference
    support_a: Vec3  # Support point on shape A
    support_b: Vec3  # Support point on shape B


class GJKSimplex:
    """Simplex for GJK algorithm."""

    def __init__(self):
        self.vertices: list[SimplexVertex] = []

    def add(self, vertex: SimplexVertex) -> None:
        """Add vertex to simplex."""
        self.vertices.append(vertex)

    def size(self) -> int:
        """Get number of vertices."""
        return len(self.vertices)

    def get_closest_points(self) -> tuple[Vec3, Vec3]:
        """Get closest points on shapes A and B."""
        if not self.vertices:
            return Vec3(), Vec3()
        if len(self.vertices) == 1:
            return self.vertices[0].support_a, self.vertices[0].support_b

        # Use barycentric coordinates for closest point
        # Simplified: use first two vertices
        v0 = self.vertices[0]
        v1 = self.vertices[1] if len(self.vertices) > 1 else v0

        # Project origin onto line segment
        ab = v1.point - v0.point
        ao = Vec3() - v0.point
        t = ao.dot(ab) / max(ab.dot(ab), 1e-10)
        t = max(0.0, min(1.0, t))

        closest_a = v0.support_a + (v1.support_a - v0.support_a) * t
        closest_b = v0.support_b + (v1.support_b - v0.support_b) * t
        return closest_a, closest_b


def _triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    """Compute (a x b) x c."""
    return b * a.dot(c) - a * b.dot(c)


def _same_direction(a: Vec3, b: Vec3) -> bool:
    """Check if vectors point in same direction."""
    return a.dot(b) > 0


def _cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product."""
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def _do_simplex_2(simplex: GJKSimplex, direction: Vec3) -> tuple[bool, Vec3]:
    """Handle 2-vertex simplex (line)."""
    b = simplex.vertices[0]
    a = simplex.vertices[1]

    ab = b.point - a.point
    ao = Vec3() - a.point

    if _same_direction(ab, ao):
        # Origin is between A and B
        new_dir = _triple_product(ab, ao, ab)
        if new_dir.length() < 1e-10:
            # Origin on line - pick perpendicular
            new_dir = _cross(ab, Vec3(1, 0, 0))
            if new_dir.length() < 1e-10:
                new_dir = _cross(ab, Vec3(0, 1, 0))
        return False, new_dir
    else:
        # Origin past A, remove B
        simplex.vertices = [a]
        return False, ao


def _do_simplex_3(simplex: GJKSimplex, direction: Vec3) -> tuple[bool, Vec3]:
    """Handle 3-vertex simplex (triangle)."""
    c = simplex.vertices[0]
    b = simplex.vertices[1]
    a = simplex.vertices[2]

    ab = b.point - a.point
    ac = c.point - a.point
    ao = Vec3() - a.point

    abc = _cross(ab, ac)

    if _same_direction(_cross(abc, ac), ao):
        if _same_direction(ac, ao):
            simplex.vertices = [c, a]
            new_dir = _triple_product(ac, ao, ac)
            # Handle zero-length direction (ao parallel to ac)
            if new_dir.length() < 1e-10:
                new_dir = _cross(ac, Vec3(1, 0, 0))
                if new_dir.length() < 1e-10:
                    new_dir = _cross(ac, Vec3(0, 1, 0))
            return False, new_dir
        else:
            simplex.vertices = [b, a]
            return _do_simplex_2(simplex, direction)
    else:
        if _same_direction(_cross(ab, abc), ao):
            simplex.vertices = [b, a]
            return _do_simplex_2(simplex, direction)
        else:
            if _same_direction(abc, ao):
                return False, abc
            else:
                simplex.vertices = [b, c, a]
                return False, abc * -1


def _do_simplex_4(simplex: GJKSimplex, direction: Vec3) -> tuple[bool, Vec3]:
    """Handle 4-vertex simplex (tetrahedron)."""
    d = simplex.vertices[0]
    c = simplex.vertices[1]
    b = simplex.vertices[2]
    a = simplex.vertices[3]

    ab = b.point - a.point
    ac = c.point - a.point
    ad = d.point - a.point
    ao = Vec3() - a.point

    abc = _cross(ab, ac)
    acd = _cross(ac, ad)
    adb = _cross(ad, ab)

    if _same_direction(abc, ao):
        simplex.vertices = [c, b, a]
        return _do_simplex_3(simplex, direction)
    if _same_direction(acd, ao):
        simplex.vertices = [d, c, a]
        return _do_simplex_3(simplex, direction)
    if _same_direction(adb, ao):
        simplex.vertices = [b, d, a]
        return _do_simplex_3(simplex, direction)

    # Origin is inside tetrahedron
    return True, direction


def _do_simplex(simplex: GJKSimplex, direction: Vec3) -> tuple[bool, Vec3]:
    """Process simplex and update search direction."""
    size = simplex.size()
    if size == 2:
        return _do_simplex_2(simplex, direction)
    elif size == 3:
        return _do_simplex_3(simplex, direction)
    elif size == 4:
        return _do_simplex_4(simplex, direction)
    return False, direction


def _support(
    shape_a: Sphere | Capsule | Box | ConvexHull,
    shape_b: Sphere | Capsule | Box | ConvexHull,
    direction: Vec3,
) -> SimplexVertex:
    """Get support point on Minkowski difference."""
    support_a = shape_a.support(direction)
    support_b = shape_b.support(direction * -1)
    return SimplexVertex(
        point=support_a - support_b,
        support_a=support_a,
        support_b=support_b,
    )


def gjk_distance(
    shape_a: Sphere | Capsule | Box | ConvexHull,
    shape_b: Sphere | Capsule | Box | ConvexHull,
    max_iterations: int = GJK_MAX_ITERATIONS,
) -> tuple[bool, float, Vec3, Vec3]:
    """
    GJK distance algorithm.

    Args:
        shape_a: First convex shape
        shape_b: Second convex shape
        max_iterations: Maximum iterations

    Returns:
        (intersecting, distance, closest_point_a, closest_point_b)
    """
    # Initial direction
    direction = Vec3(1, 0, 0)

    simplex = GJKSimplex()
    support = _support(shape_a, shape_b, direction)
    simplex.add(support)

    direction = Vec3() - support.point
    if direction.length() < 1e-10:
        direction = Vec3(1, 0, 0)

    for _ in range(max_iterations):
        support = _support(shape_a, shape_b, direction)

        if support.point.dot(direction) < 0:
            # No intersection, compute distance
            closest_a, closest_b = simplex.get_closest_points()
            distance = (closest_a - closest_b).length()
            return False, distance, closest_a, closest_b

        simplex.add(support)

        contains_origin, direction = _do_simplex(simplex, direction)
        if contains_origin:
            return True, 0.0, Vec3(), Vec3()

        if direction.length() < 1e-10:
            break

    closest_a, closest_b = simplex.get_closest_points()
    distance = (closest_a - closest_b).length()
    return distance < CONTACT_TOLERANCE, distance, closest_a, closest_b


# =============================================================================
# EPA Algorithm
# =============================================================================


@dataclass
class EPAFace:
    """Face in EPA polytope."""

    indices: tuple[int, int, int]
    normal: Vec3
    distance: float


def _compute_face_normal(
    vertices: list[SimplexVertex], face: tuple[int, int, int]
) -> tuple[Vec3, float]:
    """Compute outward-pointing normal and distance for face."""
    a = vertices[face[0]].point
    b = vertices[face[1]].point
    c = vertices[face[2]].point

    ab = b - a
    ac = c - a
    normal = _cross(ab, ac)

    length = normal.length()
    if length < 1e-10:
        return Vec3(0, 1, 0), 0.0

    normal = normal * (1.0 / length)
    distance = normal.dot(a)

    if distance < 0:
        normal = normal * -1
        distance = -distance

    return normal, distance


def epa_penetration(
    shape_a: Sphere | Capsule | Box | ConvexHull,
    shape_b: Sphere | Capsule | Box | ConvexHull,
    simplex: GJKSimplex,
    max_iterations: int = EPA_MAX_ITERATIONS,
    tolerance: float = EPA_TOLERANCE,
) -> ContactResult:
    """
    EPA algorithm for penetration depth.

    Args:
        shape_a: First convex shape
        shape_b: Second convex shape
        simplex: GJK simplex containing origin
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        ContactResult with penetration info
    """
    # Ensure we have a tetrahedron
    vertices = list(simplex.vertices)

    # Build initial faces (4 faces of tetrahedron)
    if len(vertices) < 4:
        # Expand to tetrahedron
        while len(vertices) < 4:
            # Find best direction to expand
            if len(vertices) == 1:
                directions = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)]
            elif len(vertices) == 2:
                ab = vertices[1].point - vertices[0].point
                directions = [
                    _cross(ab, Vec3(1, 0, 0)),
                    _cross(ab, Vec3(0, 1, 0)),
                ]
            else:
                ab = vertices[1].point - vertices[0].point
                ac = vertices[2].point - vertices[0].point
                directions = [_cross(ab, ac), _cross(ab, ac) * -1]

            for d in directions:
                if d.length() > 1e-10:
                    support = _support(shape_a, shape_b, d)
                    if support.point.length() > 1e-10:
                        vertices.append(support)
                        break
            else:
                # Couldn't expand - shapes might be touching at a point
                return ContactResult(
                    colliding=True,
                    normal=Vec3(0, 1, 0),
                    depth=CONTACT_TOLERANCE,
                    points=[vertices[0].support_a],
                )

    # Initial tetrahedron faces
    faces: list[EPAFace] = []
    face_indices = [
        (0, 1, 2),
        (0, 3, 1),
        (0, 2, 3),
        (1, 3, 2),
    ]

    for indices in face_indices:
        normal, distance = _compute_face_normal(vertices, indices)
        faces.append(EPAFace(indices, normal, distance))

    # EPA iteration
    for _ in range(max_iterations):
        # Find closest face to origin
        min_face = min(faces, key=lambda f: f.distance)

        # Get support point in face normal direction
        support = _support(shape_a, shape_b, min_face.normal)
        new_distance = support.point.dot(min_face.normal)

        # Check for convergence
        if new_distance - min_face.distance < tolerance:
            # Found closest face
            # Compute contact point using barycentric coordinates
            a = vertices[min_face.indices[0]]
            b = vertices[min_face.indices[1]]
            c = vertices[min_face.indices[2]]

            # Approximate: use centroid
            contact_a = (a.support_a + b.support_a + c.support_a) * (1.0 / 3.0)
            contact_b = (a.support_b + b.support_b + c.support_b) * (1.0 / 3.0)

            return ContactResult(
                colliding=True,
                normal=min_face.normal,
                depth=min_face.distance,
                points=[contact_a, contact_b],
            )

        # Add new vertex
        new_index = len(vertices)
        vertices.append(support)

        # Remove faces visible from new vertex
        new_faces: list[EPAFace] = []
        edges: list[tuple[int, int]] = []

        for face in faces:
            # Check if face is visible from new vertex
            to_vertex = support.point - vertices[face.indices[0]].point
            if face.normal.dot(to_vertex) > 0:
                # Face is visible - add edges to edge list
                for i in range(3):
                    edge = (face.indices[i], face.indices[(i + 1) % 3])
                    # Check if edge already exists (shared edge)
                    reverse = (edge[1], edge[0])
                    if reverse in edges:
                        edges.remove(reverse)
                    else:
                        edges.append(edge)
            else:
                new_faces.append(face)

        # Create new faces from edges to new vertex
        for edge in edges:
            indices = (edge[0], edge[1], new_index)
            normal, distance = _compute_face_normal(vertices, indices)
            new_faces.append(EPAFace(indices, normal, distance))

        faces = new_faces

        if not faces:
            break

    # Max iterations reached
    if faces:
        min_face = min(faces, key=lambda f: f.distance)
        return ContactResult(
            colliding=True,
            normal=min_face.normal,
            depth=min_face.distance,
            points=[vertices[min_face.indices[0]].support_a],
        )

    # Fallback: EPA failed to find a proper result (degenerate case)
    # Use the initial simplex to estimate a contact point and normal
    if vertices:
        # Estimate normal from first support point
        fallback_normal = vertices[0].point.normalized()
        if fallback_normal.length() < NUMERICAL_EPSILON:
            fallback_normal = Vec3(0, 1, 0)
        return ContactResult(
            colliding=True,
            normal=fallback_normal,
            depth=CONTACT_TOLERANCE,
            points=[vertices[0].support_a],
        )

    return ContactResult(colliding=True, normal=Vec3(0, 1, 0), depth=CONTACT_TOLERANCE)


# =============================================================================
# SAT Algorithm
# =============================================================================


def sat_test(box_a: Box, box_b: Box) -> ContactResult:
    """
    Separating Axis Theorem for OBB-OBB collision.

    Args:
        box_a: First oriented bounding box
        box_b: Second oriented bounding box

    Returns:
        ContactResult with collision info
    """
    # Get axes to test
    axes_a = list(box_a.axes)
    axes_b = list(box_b.axes)

    # 15 axes total: 3 from A, 3 from B, 9 edge pairs
    test_axes: list[Vec3] = []

    # Face normals
    test_axes.extend(axes_a)
    test_axes.extend(axes_b)

    # Edge cross products
    for axis_a in axes_a:
        for axis_b in axes_b:
            cross = _cross(axis_a, axis_b)
            if cross.length() > SAT_EDGE_BIAS:
                test_axes.append(cross.normalized())

    # Center difference
    d = box_b.center - box_a.center

    min_overlap = float("inf")
    min_axis = Vec3(1, 0, 0)

    extents_a = [box_a.half_extents.x, box_a.half_extents.y, box_a.half_extents.z]
    extents_b = [box_b.half_extents.x, box_b.half_extents.y, box_b.half_extents.z]

    for axis in test_axes:
        if axis.length() < SAT_EDGE_BIAS:
            continue

        axis = axis.normalized()

        # Project boxes onto axis
        projection_a = sum(
            extents_a[i] * abs(axes_a[i].dot(axis)) for i in range(3)
        )
        projection_b = sum(
            extents_b[i] * abs(axes_b[i].dot(axis)) for i in range(3)
        )

        distance = abs(d.dot(axis))
        overlap = projection_a + projection_b - distance

        if overlap < 0:
            # Separating axis found - no collision
            return ContactResult(colliding=False, distance=-overlap)

        if overlap < min_overlap:
            min_overlap = overlap
            min_axis = axis
            # Ensure normal points from A to B
            if d.dot(axis) < 0:
                min_axis = min_axis * -1

    # Collision detected - generate contact points
    contact_points = _generate_box_contacts(box_a, box_b, min_axis, min_overlap)

    return ContactResult(
        colliding=True,
        normal=min_axis,
        depth=min_overlap,
        points=contact_points,
    )


def _generate_box_contacts(
    box_a: Box, box_b: Box, normal: Vec3, depth: float
) -> list[Vec3]:
    """Generate contact points for box-box collision."""
    # Find reference and incident faces
    # Reference face: face most aligned with collision normal

    best_dot = float("-inf")
    ref_box = box_a
    inc_box = box_b
    ref_axis_idx = 0
    flip_normal = False

    for i, axis in enumerate(box_a.axes):
        dot = abs(normal.dot(axis))
        if dot > best_dot:
            best_dot = dot
            ref_axis_idx = i
            ref_box = box_a
            inc_box = box_b
            flip_normal = normal.dot(axis) < 0

    for i, axis in enumerate(box_b.axes):
        dot = abs(normal.dot(axis))
        if dot > best_dot:
            best_dot = dot
            ref_axis_idx = i
            ref_box = box_b
            inc_box = box_a
            flip_normal = normal.dot(axis) > 0

    # Get incident face vertices
    inc_vertices = inc_box.get_vertices()

    # Sort by distance along normal
    inc_vertices.sort(key=lambda v: -v.dot(normal) if flip_normal else v.dot(normal))

    # Take closest 4 vertices as contact candidates
    contacts = inc_vertices[:4]

    # Clip to reference face
    ref_normal = ref_box.axes[ref_axis_idx]
    if flip_normal:
        ref_normal = ref_normal * -1

    ref_extents = [
        ref_box.half_extents.x,
        ref_box.half_extents.y,
        ref_box.half_extents.z,
    ]
    ref_offset = ref_normal.dot(ref_box.center) + ref_extents[ref_axis_idx]

    # Filter points behind reference face
    result_contacts: list[Vec3] = []
    for contact in contacts:
        dist = ref_normal.dot(contact) - ref_offset
        if dist <= CONTACT_TOLERANCE:
            # Project onto reference face
            projected = contact - ref_normal * dist
            result_contacts.append(projected)

    return result_contacts[:MAX_CONTACT_POINTS]


# =============================================================================
# Specialized Shape Tests
# =============================================================================


def sphere_sphere(sphere_a: Sphere, sphere_b: Sphere) -> ContactResult:
    """
    Sphere-sphere collision test.

    Args:
        sphere_a: First sphere
        sphere_b: Second sphere

    Returns:
        ContactResult
    """
    diff = sphere_b.center - sphere_a.center
    dist_sq = diff.dot(diff)
    radius_sum = sphere_a.radius + sphere_b.radius

    if dist_sq > radius_sum * radius_sum:
        distance = math.sqrt(dist_sq) - radius_sum
        return ContactResult(colliding=False, distance=distance)

    dist = math.sqrt(dist_sq) if dist_sq > 1e-10 else 1e-10
    normal = diff * (1.0 / dist) if dist > 1e-10 else Vec3(0, 1, 0)

    depth = radius_sum - dist
    contact_point = sphere_a.center + normal * (sphere_a.radius - depth * 0.5)

    return ContactResult(
        colliding=True,
        normal=normal,
        depth=depth,
        points=[contact_point],
    )


def sphere_capsule(sphere: Sphere, capsule: Capsule) -> ContactResult:
    """
    Sphere-capsule collision test.

    Args:
        sphere: Sphere shape
        capsule: Capsule shape

    Returns:
        ContactResult
    """
    # Find closest point on capsule axis
    closest = capsule.closest_point_on_axis(sphere.center)

    diff = sphere.center - closest
    dist_sq = diff.dot(diff)
    radius_sum = sphere.radius + capsule.radius

    if dist_sq > radius_sum * radius_sum:
        distance = math.sqrt(dist_sq) - radius_sum
        return ContactResult(colliding=False, distance=distance)

    dist = math.sqrt(dist_sq) if dist_sq > 1e-10 else 1e-10
    normal = diff * (1.0 / dist) if dist > 1e-10 else Vec3(0, 1, 0)

    depth = radius_sum - dist
    contact_point = closest + normal * (capsule.radius - depth * 0.5)

    return ContactResult(
        colliding=True,
        normal=normal,
        depth=depth,
        points=[contact_point],
    )


def capsule_capsule(capsule_a: Capsule, capsule_b: Capsule) -> ContactResult:
    """
    Capsule-capsule collision test.

    Args:
        capsule_a: First capsule
        capsule_b: Second capsule

    Returns:
        ContactResult
    """
    # Find closest points between line segments
    p1, p2 = capsule_a.start, capsule_a.end
    p3, p4 = capsule_b.start, capsule_b.end

    d1 = p2 - p1  # Direction of segment 1
    d2 = p4 - p3  # Direction of segment 2
    r = p1 - p3

    a = d1.dot(d1)
    e = d2.dot(d2)
    f = d2.dot(r)

    # Check if both segments degenerate to points
    if a < 1e-10 and e < 1e-10:
        closest_a = p1
        closest_b = p3
    elif a < 1e-10:
        # First segment is a point
        closest_a = p1
        t = max(0.0, min(1.0, f / e))
        closest_b = p3 + d2 * t
    elif e < 1e-10:
        # Second segment is a point
        closest_b = p3
        s = max(0.0, min(1.0, -r.dot(d1) / a))
        closest_a = p1 + d1 * s
    else:
        c = d1.dot(r)
        b = d1.dot(d2)
        denom = a * e - b * b

        if denom > 1e-10:
            s = max(0.0, min(1.0, (b * f - c * e) / denom))
        else:
            s = 0.0

        t = (b * s + f) / e
        if t < 0.0:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        elif t > 1.0:
            t = 1.0
            s = max(0.0, min(1.0, (b - c) / a))

        closest_a = p1 + d1 * s
        closest_b = p3 + d2 * t

    diff = closest_a - closest_b
    dist_sq = diff.dot(diff)
    radius_sum = capsule_a.radius + capsule_b.radius

    if dist_sq > radius_sum * radius_sum:
        distance = math.sqrt(dist_sq) - radius_sum
        return ContactResult(colliding=False, distance=distance)

    dist = math.sqrt(dist_sq) if dist_sq > 1e-10 else 1e-10
    normal = diff * (1.0 / dist) if dist > 1e-10 else Vec3(0, 1, 0)

    depth = radius_sum - dist
    contact_point = closest_b + normal * (capsule_b.radius - depth * 0.5)

    return ContactResult(
        colliding=True,
        normal=normal,
        depth=depth,
        points=[contact_point],
    )


def box_box(box_a: Box, box_b: Box) -> ContactResult:
    """
    Box-box collision using SAT.

    Args:
        box_a: First box
        box_b: Second box

    Returns:
        ContactResult
    """
    return sat_test(box_a, box_b)


def sphere_box(sphere: Sphere, box: Box) -> ContactResult:
    """
    Sphere-box collision test.

    Args:
        sphere: Sphere shape
        box: Box shape

    Returns:
        ContactResult
    """
    # Transform sphere center to box local space
    local_center = sphere.center - box.center

    # Find closest point on box
    closest = Vec3()
    for i, axis in enumerate(box.axes):
        extent = [box.half_extents.x, box.half_extents.y, box.half_extents.z][i]
        dist = local_center.dot(axis)
        dist = max(-extent, min(extent, dist))
        closest = closest + axis * dist

    closest = closest + box.center

    diff = sphere.center - closest
    dist_sq = diff.dot(diff)

    if dist_sq > sphere.radius * sphere.radius:
        distance = math.sqrt(dist_sq) - sphere.radius
        return ContactResult(colliding=False, distance=distance)

    dist = math.sqrt(dist_sq) if dist_sq > 1e-10 else 1e-10
    normal = diff * (1.0 / dist) if dist > 1e-10 else Vec3(0, 1, 0)

    depth = sphere.radius - dist
    contact_point = closest

    return ContactResult(
        colliding=True,
        normal=normal,
        depth=depth,
        points=[contact_point],
    )


def capsule_box(capsule: Capsule, box: Box) -> ContactResult:
    """
    Capsule-box collision test.

    Args:
        capsule: Capsule shape
        box: Box shape

    Returns:
        ContactResult using GJK/EPA
    """
    # Use GJK for general case
    intersecting, distance, closest_a, closest_b = gjk_distance(capsule, box)

    if not intersecting:
        return ContactResult(colliding=False, distance=distance)

    # Build simplex for EPA
    simplex = GJKSimplex()
    direction = Vec3(1, 0, 0)
    support = _support(capsule, box, direction)
    simplex.add(support)

    # Need to build full simplex
    directions = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(-1, -1, -1)]
    for d in directions:
        support = _support(capsule, box, d)
        simplex.add(support)
        if simplex.size() >= 4:
            break

    return epa_penetration(capsule, box, simplex)


# =============================================================================
# Generic Collision Test
# =============================================================================


def collide_shapes(
    shape_a: Sphere | Capsule | Box | ConvexHull,
    shape_b: Sphere | Capsule | Box | ConvexHull,
    algorithm: NarrowphaseAlgorithm = NarrowphaseAlgorithm.GJK_EPA,
) -> ContactResult:
    """
    Generic collision test between two shapes.

    Args:
        shape_a: First shape
        shape_b: Second shape
        algorithm: Algorithm to use

    Returns:
        ContactResult
    """
    # Use specialized tests when available
    if isinstance(shape_a, Sphere) and isinstance(shape_b, Sphere):
        return sphere_sphere(shape_a, shape_b)
    if isinstance(shape_a, Sphere) and isinstance(shape_b, Capsule):
        return sphere_capsule(shape_a, shape_b)
    if isinstance(shape_a, Capsule) and isinstance(shape_b, Sphere):
        result = sphere_capsule(shape_b, shape_a)
        result.normal = result.normal * -1
        return result
    if isinstance(shape_a, Capsule) and isinstance(shape_b, Capsule):
        return capsule_capsule(shape_a, shape_b)
    if isinstance(shape_a, Box) and isinstance(shape_b, Box):
        if algorithm == NarrowphaseAlgorithm.SAT:
            return sat_test(shape_a, shape_b)
        return box_box(shape_a, shape_b)
    if isinstance(shape_a, Sphere) and isinstance(shape_b, Box):
        return sphere_box(shape_a, shape_b)
    if isinstance(shape_a, Box) and isinstance(shape_b, Sphere):
        result = sphere_box(shape_b, shape_a)
        result.normal = result.normal * -1
        return result
    if isinstance(shape_a, Capsule) and isinstance(shape_b, Box):
        return capsule_box(shape_a, shape_b)
    if isinstance(shape_a, Box) and isinstance(shape_b, Capsule):
        result = capsule_box(shape_b, shape_a)
        result.normal = result.normal * -1
        return result

    # General GJK/EPA for convex shapes
    intersecting, distance, closest_a, closest_b = gjk_distance(shape_a, shape_b)

    if not intersecting:
        return ContactResult(colliding=False, distance=distance)

    # Build simplex for EPA
    simplex = GJKSimplex()
    directions = [Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(-1, -1, -1)]
    for d in directions:
        support = _support(shape_a, shape_b, d)
        simplex.add(support)
        if simplex.size() >= 4:
            break

    return epa_penetration(shape_a, shape_b, simplex)
