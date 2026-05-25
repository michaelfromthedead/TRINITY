"""
Descriptor composition engine.

Provides safe composition of descriptors with validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeVar

from trinity.descriptors.base import BaseDescriptor

if TYPE_CHECKING:
    from trinity.decorators.ops import Step

T = TypeVar("T")


class DescriptorCompositionError(TypeError):
    """Raised when descriptor composition is invalid."""

    pass


class DescriptorComposer:
    """
    Composes descriptors into chains with validation.

    Usage:
        composer = DescriptorComposer()
        chain = DescriptorComposer.compose(
            NetworkedDescriptor(field_type=float),
            TrackedDescriptor(field_type=float),
            ValidatedDescriptor(field_type=float),
            StorageDescriptor(field_type=float, default=100),
        )
        # Returns: Networked -> Tracked -> Validated -> Storage
    """

    @staticmethod
    def compose(*descriptors: BaseDescriptor[T]) -> BaseDescriptor[T]:
        """
        Compose descriptors into a chain.

        Args:
            *descriptors: Descriptors in order from outermost to innermost.

        Returns:
            The outermost descriptor, wrapping all others.

        Raises:
            DescriptorCompositionError: If composition is invalid.
        """
        if not descriptors:
            raise DescriptorCompositionError("Cannot compose empty descriptor list")

        if len(descriptors) == 1:
            return descriptors[0]

        # Validate the full chain first
        DescriptorComposer._validate_chain(descriptors)

        # Build chain from innermost to outermost
        reversed_descriptors = list(reversed(descriptors))
        current = reversed_descriptors[0]

        for outer in reversed_descriptors[1:]:
            outer._inner = current
            current = outer

        return current

    @staticmethod
    def _validate_chain(descriptors: tuple[BaseDescriptor[T], ...]) -> None:
        """Validate that descriptors can be composed in the given order."""

        # Check for exclusions (global conflicts)
        all_ids = {d.descriptor_id for d in descriptors}
        for desc in descriptors:
            conflicts = all_ids & set(desc.excludes)
            if conflicts:
                raise DescriptorCompositionError(
                    f"Descriptor '{desc.descriptor_id}' cannot coexist with: {conflicts}"
                )

        # Check pairwise compatibility (outer to inner)
        for i in range(len(descriptors) - 1):
            outer = descriptors[i]
            inner = descriptors[i + 1]

            # Check if outer accepts inner
            outer_accepts = outer.accepts_inner
            if outer_accepts and outer_accepts != ("*",):
                if inner.descriptor_id not in outer_accepts:
                    raise DescriptorCompositionError(
                        f"'{outer.descriptor_id}' cannot wrap '{inner.descriptor_id}'. "
                        f"Accepts: {outer_accepts}"
                    )

            # Check if inner accepts being wrapped by outer
            inner_accepts = inner.accepts_outer
            if inner_accepts and inner_accepts != ("*",):
                if outer.descriptor_id not in inner_accepts:
                    raise DescriptorCompositionError(
                        f"'{inner.descriptor_id}' cannot be wrapped by '{outer.descriptor_id}'. "
                        f"Accepts outer: {inner_accepts}"
                    )

    @staticmethod
    def can_compose(*descriptors: BaseDescriptor[T]) -> tuple[bool, Optional[str]]:
        """
        Check if descriptors can be composed without raising.

        Returns:
            (True, None) if valid, (False, error_message) if invalid.
        """
        try:
            DescriptorComposer._validate_chain(tuple(descriptors))
            return (True, None)
        except DescriptorCompositionError as e:
            return (False, str(e))

    @staticmethod
    def collect_steps(descriptor: BaseDescriptor[T]) -> list:
        """Collect Steps from entire descriptor chain."""
        all_steps: list = []
        for desc in descriptor.get_chain():
            all_steps.extend(getattr(desc, "descriptor_steps", []))
        return all_steps

    @staticmethod
    def explain_chain(descriptor: BaseDescriptor[T]) -> str:
        """
        Generate a human-readable explanation of a descriptor chain.

        Args:
            descriptor: The outermost descriptor in the chain.

        Returns:
            A formatted string explaining the chain.
        """
        chain = descriptor.get_chain()
        lines = [f"Descriptor chain for '{descriptor.name}':"]

        for i, desc in enumerate(chain):
            indent = "  " * i
            arrow = "→ " if i > 0 else ""
            meta = desc.get_metadata()

            # Format descriptor info
            info_parts = [f"{arrow}{desc.descriptor_id}"]

            # Add relevant metadata
            if "range" in meta:
                info_parts.append(f"range={meta['range']}")
            if meta.get("network"):
                net = meta["network"]
                info_parts.append(f"authority={net['authority']}")
            if meta.get("ttl"):
                info_parts.append(f"ttl={meta['ttl']}s")

            lines.append(f"{indent}{' '.join(info_parts)}")

        return "\n".join(lines)
