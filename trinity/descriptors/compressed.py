"""
CompressedDescriptor — compress/decompress values on set/get.
"""

from __future__ import annotations

import zlib
from typing import Any, Optional

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

__all__ = ["CompressedDescriptor"]


_MAX_DECOMPRESS_SIZE: int = 100 * 1024 * 1024  # 100 MB


class CompressedDescriptor(BaseDescriptor[bytes]):
    """Descriptor that compresses values on write and decompresses on read."""

    __slots__ = ("_algorithm",)

    descriptor_id: str = "compressed"

    def __init__(
        self,
        algorithm: str = "zlib",
        field_type: type = bytes,
        inner: Optional[BaseDescriptor] = None,
        **config: Any,
    ) -> None:
        if algorithm not in ("zlib", "lz4", "none"):
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        super().__init__(field_type=field_type, inner=inner, **config)
        self._algorithm = algorithm

    # ------------------------------------------------------------------
    # Core descriptor overrides
    # ------------------------------------------------------------------

    def pre_set(self, obj: Any, value: Any) -> bytes:
        """Compress the value before storage."""
        return self._compress(value)

    def post_get(self, obj: Any, value: Any) -> Any:
        """Decompress the value after retrieval."""
        if value is None:
            return None
        return self._decompress(value)

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    def _compress(self, value: Any) -> bytes:
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, bytes):
            value = str(value).encode("utf-8")
        if self._algorithm == "zlib":
            return zlib.compress(value)
        if self._algorithm == "lz4":
            try:
                import lz4.frame  # type: ignore[import-untyped]
                return lz4.frame.compress(value)
            except ImportError:
                raise ImportError("lz4 package required for lz4 compression")
        # algorithm == "none"
        return value

    def _decompress(self, data: bytes) -> bytes:
        if self._algorithm == "zlib":
            result = zlib.decompress(data, zlib.MAX_WBITS, _MAX_DECOMPRESS_SIZE)
            if len(result) > _MAX_DECOMPRESS_SIZE:
                raise ValueError(
                    f"Decompressed size {len(result)} exceeds limit "
                    f"of {_MAX_DECOMPRESS_SIZE} bytes"
                )
            return result
        if self._algorithm == "lz4":
            try:
                import lz4.frame  # type: ignore[import-untyped]
                result = lz4.frame.decompress(data)
                if len(result) > _MAX_DECOMPRESS_SIZE:
                    raise ValueError(
                        f"Decompressed size {len(result)} exceeds limit "
                        f"of {_MAX_DECOMPRESS_SIZE} bytes"
                    )
                return result
            except ImportError:
                raise ImportError("lz4 package required for lz4 decompression")
        return data

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.INTERCEPT, {"get": "decompress", "set": "compress"}),
            Step(Op.TAG, {"key": "compressed", "value": self._algorithm}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["algorithm"] = self._algorithm
        return meta
