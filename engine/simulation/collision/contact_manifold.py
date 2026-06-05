"""
Contact Manifold Management.

This module handles contact point persistence, reduction, and caching
for physics simulations. Contact manifolds maintain sets of contact
points between colliding shape pairs with support for:
- Persistent contact caching
- Warm starting
- Manifold reduction to optimal contact set
- Age-based contact removal
"""

from dataclasses import dataclass, field
from typing import Callable
import math

from .broadphase import Vec3
from .config import (
    MAX_CONTACT_POINTS,
    CONTACT_MAX_AGE,
    CONTACT_MATCH_THRESHOLD,
    WARM_START_FACTOR,
    CONTACT_TOLERANCE,
    PARALLEL_THRESHOLD,
)


# =============================================================================
# Contact Point
# =============================================================================


@dataclass
class ContactPoint:
    """Single contact point in a manifold."""

    # Position in world space (on body A)
    position: Vec3 = field(default_factory=Vec3)

    # Contact normal (from A to B)
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))

    # Penetration depth (positive = overlapping)
    depth: float = 0.0

    # Material properties
    friction: float = 0.5
    restitution: float = 0.0

    # Local positions for persistence
    local_position_a: Vec3 = field(default_factory=Vec3)
    local_position_b: Vec3 = field(default_factory=Vec3)

    # Cached impulses for warm starting
    normal_impulse: float = 0.0
    tangent_impulse_1: float = 0.0
    tangent_impulse_2: float = 0.0

    # Contact age (frames since creation)
    age: int = 0

    # Feature IDs for persistent contact matching
    feature_id_a: int = -1
    feature_id_b: int = -1

    # Unique contact ID
    contact_id: int = -1

    def distance_to(self, other: "ContactPoint") -> float:
        """Calculate distance to another contact point."""
        diff = self.position - other.position
        return diff.length()

    def update_impulse(
        self,
        normal_impulse: float,
        tangent_impulse_1: float = 0.0,
        tangent_impulse_2: float = 0.0,
    ) -> None:
        """Update cached impulses."""
        self.normal_impulse = normal_impulse
        self.tangent_impulse_1 = tangent_impulse_1
        self.tangent_impulse_2 = tangent_impulse_2

    def get_warm_start_impulse(self) -> tuple[float, float, float]:
        """Get warm starting impulses scaled by retention factor."""
        return (
            self.normal_impulse * WARM_START_FACTOR,
            self.tangent_impulse_1 * WARM_START_FACTOR,
            self.tangent_impulse_2 * WARM_START_FACTOR,
        )


# =============================================================================
# Contact Manifold
# =============================================================================


@dataclass
class ManifoldKey:
    """Key for identifying manifold between two bodies."""

    body_a: int
    body_b: int

    def __hash__(self) -> int:
        # Order-independent hash
        return hash((min(self.body_a, self.body_b), max(self.body_a, self.body_b)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ManifoldKey):
            return False
        return (self.body_a == other.body_a and self.body_b == other.body_b) or (
            self.body_a == other.body_b and self.body_b == other.body_a
        )


class ContactManifold:
    """
    Manages a set of contact points between two bodies.

    Features:
    - Persistent contact caching with warm starting
    - Automatic manifold reduction to optimal contact set
    - Age-based contact removal
    - Support for contact callbacks
    """

    def __init__(
        self,
        body_a: int,
        body_b: int,
        max_contacts: int = MAX_CONTACT_POINTS,
        match_threshold: float = CONTACT_MATCH_THRESHOLD,
    ):
        self._body_a = body_a
        self._body_b = body_b
        self._max_contacts = max_contacts
        self._match_threshold = match_threshold
        self._contacts: list[ContactPoint] = []
        self._next_contact_id = 0
        self._is_touching = False
        self._was_touching = False

    @property
    def body_a(self) -> int:
        """Get first body ID."""
        return self._body_a

    @property
    def body_b(self) -> int:
        """Get second body ID."""
        return self._body_b

    @property
    def contact_count(self) -> int:
        """Get number of contacts."""
        return len(self._contacts)

    @property
    def contacts(self) -> list[ContactPoint]:
        """Get contact points (read-only copy)."""
        return list(self._contacts)

    @property
    def is_touching(self) -> bool:
        """Check if bodies are currently in contact."""
        return self._is_touching

    @property
    def was_touching(self) -> bool:
        """Check if bodies were touching last frame."""
        return self._was_touching

    def get_key(self) -> ManifoldKey:
        """Get manifold key for this pair."""
        return ManifoldKey(self._body_a, self._body_b)

    def add_contact(
        self,
        position: Vec3,
        normal: Vec3,
        depth: float,
        friction: float = 0.5,
        restitution: float = 0.0,
        local_a: Vec3 | None = None,
        local_b: Vec3 | None = None,
        feature_id_a: int = -1,
        feature_id_b: int = -1,
    ) -> ContactPoint:
        """
        Add a new contact point to the manifold.

        If a matching contact exists, update it instead of creating new.
        If manifold is full, reduce to best contacts.

        Args:
            position: World position of contact
            normal: Contact normal (A to B)
            depth: Penetration depth
            friction: Friction coefficient
            restitution: Restitution coefficient
            local_a: Local position on body A
            local_b: Local position on body B
            feature_id_a: Feature ID on shape A
            feature_id_b: Feature ID on shape B

        Returns:
            The added or updated contact point
        """
        # Check for matching existing contact
        for contact in self._contacts:
            if self._contacts_match(
                contact, position, feature_id_a, feature_id_b
            ):
                # Update existing contact
                contact.position = position
                contact.normal = normal
                contact.depth = depth
                contact.friction = friction
                contact.restitution = restitution
                if local_a:
                    contact.local_position_a = local_a
                if local_b:
                    contact.local_position_b = local_b
                contact.age = 0  # Reset age
                return contact

        # Create new contact
        contact = ContactPoint(
            position=position,
            normal=normal,
            depth=depth,
            friction=friction,
            restitution=restitution,
            local_position_a=local_a or Vec3(),
            local_position_b=local_b or Vec3(),
            feature_id_a=feature_id_a,
            feature_id_b=feature_id_b,
            contact_id=self._next_contact_id,
            age=0,
        )
        self._next_contact_id += 1
        self._contacts.append(contact)

        # Reduce if over capacity
        if len(self._contacts) > self._max_contacts:
            self.reduce_manifold()

        return contact

    def remove_contact(self, contact_id: int) -> bool:
        """
        Remove a specific contact by ID.

        Args:
            contact_id: ID of contact to remove

        Returns:
            True if contact was removed
        """
        for i, contact in enumerate(self._contacts):
            if contact.contact_id == contact_id:
                self._contacts.pop(i)
                return True
        return False

    def _contacts_match(
        self,
        contact: ContactPoint,
        position: Vec3,
        feature_id_a: int,
        feature_id_b: int,
    ) -> bool:
        """Check if contact matches new contact data."""
        # Feature ID matching (if available)
        if feature_id_a >= 0 and feature_id_b >= 0:
            if (
                contact.feature_id_a == feature_id_a
                and contact.feature_id_b == feature_id_b
            ):
                return True

        # Position-based matching
        diff = contact.position - position
        return diff.length() < self._match_threshold

    def reduce_manifold(self) -> None:
        """
        Reduce manifold to optimal set of contacts.

        Uses a greedy algorithm to select contacts that maximize
        the contact area coverage.
        """
        if len(self._contacts) <= self._max_contacts:
            return

        # Keep deepest contact
        self._contacts.sort(key=lambda c: -c.depth)
        kept: list[ContactPoint] = [self._contacts[0]]

        # Add contacts that maximize spread
        remaining = self._contacts[1:]

        while len(kept) < self._max_contacts and remaining:
            best_contact = None
            best_min_dist = -1.0

            for contact in remaining:
                # Find minimum distance to already kept contacts
                min_dist = min(
                    contact.distance_to(k) for k in kept
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_contact = contact

            if best_contact:
                kept.append(best_contact)
                remaining.remove(best_contact)
            else:
                break

        self._contacts = kept

    def age_contacts(self) -> list[ContactPoint]:
        """
        Age all contacts and remove old ones.

        Returns:
            List of removed contacts
        """
        removed: list[ContactPoint] = []

        for contact in self._contacts[:]:
            contact.age += 1
            if contact.age > CONTACT_MAX_AGE:
                self._contacts.remove(contact)
                removed.append(contact)

        return removed

    def refresh_contacts(
        self,
        transform_a: Callable[[Vec3], Vec3],
        transform_b: Callable[[Vec3], Vec3],
    ) -> list[ContactPoint]:
        """
        Refresh contacts using current body transforms.

        Removes contacts that have separated beyond tolerance.

        Args:
            transform_a: Transform for body A (local to world)
            transform_b: Transform for body B (local to world)

        Returns:
            List of removed contacts
        """
        removed: list[ContactPoint] = []

        for contact in self._contacts[:]:
            # Recompute world positions from local
            world_a = transform_a(contact.local_position_a)
            world_b = transform_b(contact.local_position_b)

            # Calculate how far the contact points have moved apart
            # We compare to the original position to detect if bodies have separated
            diff = world_b - world_a

            # Project the difference onto the normal to get separation along contact direction
            separation_along_normal = diff.dot(contact.normal)

            # Calculate lateral movement (perpendicular to normal)
            lateral_diff = diff - contact.normal * separation_along_normal
            lateral_distance = lateral_diff.length()

            # Contact is invalid if:
            # 1. Separation along normal exceeds depth + tolerance (bodies moved apart)
            # 2. Lateral drift exceeds tolerance (contact point slid away)
            depth_threshold = contact.depth + CONTACT_TOLERANCE

            if separation_along_normal > depth_threshold or lateral_distance > CONTACT_TOLERANCE:
                # Contact has separated
                self._contacts.remove(contact)
                removed.append(contact)
            else:
                # Update contact
                contact.position = (world_a + world_b) * 0.5
                contact.depth = -separation_along_normal

        return removed

    def clear(self) -> None:
        """Remove all contacts."""
        self._contacts.clear()

    def update_touching_state(self) -> tuple[bool, bool, bool]:
        """
        Update touching state.

        Returns:
            (began, persist, ended) - contact state changes
        """
        self._was_touching = self._is_touching
        self._is_touching = len(self._contacts) > 0

        began = self._is_touching and not self._was_touching
        persist = self._is_touching and self._was_touching
        ended = not self._is_touching and self._was_touching

        return began, persist, ended

    def get_average_normal(self) -> Vec3:
        """Get average contact normal."""
        if not self._contacts:
            return Vec3(0, 1, 0)

        total = Vec3()
        for contact in self._contacts:
            total = total + contact.normal
        return total.normalized()

    def get_average_position(self) -> Vec3:
        """Get average contact position."""
        if not self._contacts:
            return Vec3()

        total = Vec3()
        for contact in self._contacts:
            total = total + contact.position
        return total * (1.0 / len(self._contacts))

    def get_max_depth(self) -> float:
        """Get maximum penetration depth."""
        if not self._contacts:
            return 0.0
        return max(c.depth for c in self._contacts)

    def get_total_impulse(self) -> float:
        """Get total normal impulse applied."""
        return sum(c.normal_impulse for c in self._contacts)


# =============================================================================
# Manifold Cache
# =============================================================================


class ManifoldCache:
    """
    Cache for contact manifolds between body pairs.

    Provides persistent contact tracking across frames with
    warm starting support.
    """

    def __init__(self, max_manifolds: int = 1024):
        self._manifolds: dict[ManifoldKey, ContactManifold] = {}
        self._max_manifolds = max_manifolds
        self._frame_count = 0

    @property
    def manifold_count(self) -> int:
        """Get number of cached manifolds."""
        return len(self._manifolds)

    def get_or_create(
        self,
        body_a: int,
        body_b: int,
        max_contacts: int = MAX_CONTACT_POINTS,
    ) -> ContactManifold:
        """
        Get existing manifold or create new one.

        Args:
            body_a: First body ID
            body_b: Second body ID
            max_contacts: Maximum contacts per manifold

        Returns:
            Contact manifold for the pair
        """
        key = ManifoldKey(body_a, body_b)

        if key in self._manifolds:
            return self._manifolds[key]

        # Create new manifold
        manifold = ContactManifold(body_a, body_b, max_contacts)
        self._manifolds[key] = manifold

        # Evict old manifolds if necessary
        if len(self._manifolds) > self._max_manifolds:
            self._evict_oldest()

        return manifold

    def get(self, body_a: int, body_b: int) -> ContactManifold | None:
        """
        Get existing manifold if it exists.

        Args:
            body_a: First body ID
            body_b: Second body ID

        Returns:
            Contact manifold or None
        """
        key = ManifoldKey(body_a, body_b)
        return self._manifolds.get(key)

    def remove(self, body_a: int, body_b: int) -> bool:
        """
        Remove manifold for a body pair.

        Args:
            body_a: First body ID
            body_b: Second body ID

        Returns:
            True if manifold was removed
        """
        key = ManifoldKey(body_a, body_b)
        if key in self._manifolds:
            del self._manifolds[key]
            return True
        return False

    def remove_body(self, body_id: int) -> int:
        """
        Remove all manifolds involving a body.

        Args:
            body_id: Body ID to remove

        Returns:
            Number of manifolds removed
        """
        to_remove: list[ManifoldKey] = []

        for key in self._manifolds:
            if key.body_a == body_id or key.body_b == body_id:
                to_remove.append(key)

        for key in to_remove:
            del self._manifolds[key]

        return len(to_remove)

    def update_frame(self) -> tuple[list[ContactManifold], list[ContactManifold]]:
        """
        Update all manifolds for new frame.

        Ages contacts and removes empty manifolds.

        Returns:
            (began_contacts, ended_contacts) - manifolds with state changes
        """
        self._frame_count += 1
        began: list[ContactManifold] = []
        ended: list[ContactManifold] = []
        to_remove: list[ManifoldKey] = []

        for key, manifold in self._manifolds.items():
            manifold.age_contacts()
            b, p, e = manifold.update_touching_state()

            if b:
                began.append(manifold)
            if e:
                ended.append(manifold)

            # Remove empty manifolds that have been empty for a while
            if manifold.contact_count == 0:
                to_remove.append(key)

        for key in to_remove:
            del self._manifolds[key]

        return began, ended

    def clear(self) -> None:
        """Clear all manifolds."""
        self._manifolds.clear()

    def get_all_manifolds(self) -> list[ContactManifold]:
        """Get all cached manifolds."""
        return list(self._manifolds.values())

    def get_touching_manifolds(self) -> list[ContactManifold]:
        """Get all manifolds with active contacts."""
        return [m for m in self._manifolds.values() if m.is_touching]

    def _evict_oldest(self) -> None:
        """Evict oldest empty manifolds."""
        # First try to remove empty manifolds
        empty_keys = [
            k for k, m in self._manifolds.items() if m.contact_count == 0
        ]

        for key in empty_keys[: len(self._manifolds) - self._max_manifolds + 1]:
            del self._manifolds[key]


# =============================================================================
# Contact Pair
# =============================================================================


@dataclass
class ContactPair:
    """
    Lightweight contact pair for constraint solver.

    Contains pre-computed data for efficient constraint solving.
    """

    body_a: int
    body_b: int
    contact: ContactPoint

    # Cached solver data
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    tangent_1: Vec3 = field(default_factory=lambda: Vec3(1, 0, 0))
    tangent_2: Vec3 = field(default_factory=lambda: Vec3(0, 0, 1))

    # Effective mass
    normal_mass: float = 0.0
    tangent_mass_1: float = 0.0
    tangent_mass_2: float = 0.0

    # Bias for position correction
    bias: float = 0.0
    restitution_bias: float = 0.0

    def compute_tangent_basis(self) -> None:
        """Compute tangent vectors orthogonal to normal."""
        # Find least-aligned axis
        if abs(self.normal.x) < PARALLEL_THRESHOLD:
            ref = Vec3(1, 0, 0)
        else:
            ref = Vec3(0, 1, 0)

        # Gram-Schmidt
        self.tangent_1 = Vec3(
            ref.y * self.normal.z - ref.z * self.normal.y,
            ref.z * self.normal.x - ref.x * self.normal.z,
            ref.x * self.normal.y - ref.y * self.normal.x,
        ).normalized()

        self.tangent_2 = Vec3(
            self.normal.y * self.tangent_1.z - self.normal.z * self.tangent_1.y,
            self.normal.z * self.tangent_1.x - self.normal.x * self.tangent_1.z,
            self.normal.x * self.tangent_1.y - self.normal.y * self.tangent_1.x,
        ).normalized()


def create_contact_pairs(manifold: ContactManifold) -> list[ContactPair]:
    """
    Create contact pairs from a manifold for solver.

    Args:
        manifold: Contact manifold

    Returns:
        List of contact pairs with pre-computed data
    """
    pairs: list[ContactPair] = []

    for contact in manifold.contacts:
        pair = ContactPair(
            body_a=manifold.body_a,
            body_b=manifold.body_b,
            contact=contact,
            normal=contact.normal,
        )
        pair.compute_tangent_basis()
        pairs.append(pair)

    return pairs
