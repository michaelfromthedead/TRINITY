"""Two-Level Segregated Fit allocator — O(1) alloc/free with low fragmentation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.core.constants import TLSF_SL_COUNT, TLSF_SL_BITS, TLSF_MIN_BLOCK
from engine.core.memory.allocator import Allocator

logger = logging.getLogger(__name__)

# Second-level subdivision count (must be power of 2)
_SL_COUNT = TLSF_SL_COUNT
_SL_BITS = TLSF_SL_BITS
_MIN_BLOCK = TLSF_MIN_BLOCK


@dataclass(slots=True)
class _Block:
    """Free block metadata."""
    offset: int
    size: int


class TLSFAllocator(Allocator):
    """Two-Level Segregated Fit general-purpose real-time allocator.

    First level: power-of-two size classes.
    Second level: ``_SL_COUNT`` linear subdivisions within each first-level class.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < _MIN_BLOCK:
            raise ValueError(f"capacity must be >= {_MIN_BLOCK}")
        self._capacity = capacity
        self._buffer = bytearray(capacity)
        self._used = 0
        self._alloc_sizes: dict[int, int] = {}

        # Compute number of first-level classes
        self._fl_count = max(1, capacity.bit_length())

        # free_lists[(fl, sl)] -> list of free blocks
        self._free_lists: Dict[Tuple[int, int], List[_Block]] = {}
        # Bitmaps for O(1) non-empty list lookup
        self._fl_bitmap = 0
        self._sl_bitmaps: Dict[int, int] = {}

        # Seed with one big free block
        self._insert_free(_Block(offset=0, size=capacity))

    # -- Allocator interface --------------------------------------------------

    def allocate(self, size: int) -> int:
        if size <= 0:
            raise ValueError("size must be positive")
        adjusted = max(size, _MIN_BLOCK)
        fl, sl = self._mapping(adjusted)

        block = self._find_suitable(fl, sl)
        if block is None:
            raise MemoryError(
                f"TLSFAllocator: cannot satisfy {size} bytes "
                f"({self._capacity - self._used} free)"
            )

        # Remove from free list
        self._remove_free(block)

        # Split if remainder is large enough
        remainder = block.size - adjusted
        if remainder >= _MIN_BLOCK:
            split = _Block(offset=block.offset + adjusted, size=remainder)
            self._insert_free(split)
            block.size = adjusted

        self._used += block.size
        self._alloc_sizes[block.offset] = block.size
        logger.debug("tlsf alloc size=%d offset=%d", block.size, block.offset)
        return block.offset

    def free(self, offset: int, size: int = 0) -> None:
        """Free bytes starting at *offset*. Size is looked up automatically if not provided."""
        if size <= 0:
            size = self._alloc_sizes.pop(offset, 0)
            if size <= 0:
                raise ValueError("size must be positive and offset was not found in tracked allocations")
        else:
            self._alloc_sizes.pop(offset, None)
        freed = _Block(offset=offset, size=size)
        self._used = max(0, self._used - size)
        self._insert_free(freed)
        self._coalesce(freed)
        logger.debug("tlsf free offset=%d size=%d", offset, size)

    def reset(self) -> None:
        self._used = 0
        self._alloc_sizes.clear()
        self._free_lists.clear()
        self._fl_bitmap = 0
        self._sl_bitmaps.clear()
        self._insert_free(_Block(offset=0, size=self._capacity))

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def capacity(self) -> int:
        return self._capacity

    # -- Internal mapping -----------------------------------------------------

    @staticmethod
    def _mapping(size: int) -> Tuple[int, int]:
        """Map *size* to (first_level, second_level) indices."""
        if size < _MIN_BLOCK:
            return 0, 0
        fl = size.bit_length() - 1
        remainder = size - (1 << fl)
        sl = (remainder >> max(0, fl - _SL_BITS)) & (_SL_COUNT - 1)
        return fl, sl

    def _find_suitable(self, fl: int, sl: int) -> Optional[_Block]:
        """Find a free block at or above the requested (fl, sl)."""
        # Search current fl from sl upward
        sl_map = self._sl_bitmaps.get(fl, 0) & (~0 << sl)
        if sl_map:
            found_sl = (sl_map & -sl_map).bit_length() - 1
            blocks = self._free_lists.get((fl, found_sl))
            if blocks:
                return blocks[0]

        # Search higher fl
        fl_map = self._fl_bitmap & (~0 << (fl + 1))
        if not fl_map:
            return None
        found_fl = (fl_map & -fl_map).bit_length() - 1
        sl_map2 = self._sl_bitmaps.get(found_fl, 0)
        if not sl_map2:
            return None
        found_sl = (sl_map2 & -sl_map2).bit_length() - 1
        blocks = self._free_lists.get((found_fl, found_sl))
        if blocks:
            return blocks[0]
        return None

    def _insert_free(self, block: _Block) -> None:
        fl, sl = self._mapping(block.size)
        key = (fl, sl)
        lst = self._free_lists.setdefault(key, [])
        lst.append(block)
        self._fl_bitmap |= 1 << fl
        self._sl_bitmaps[fl] = self._sl_bitmaps.get(fl, 0) | (1 << sl)

    def _remove_free(self, block: _Block) -> None:
        fl, sl = self._mapping(block.size)
        key = (fl, sl)
        lst = self._free_lists.get(key, [])
        if block in lst:
            lst.remove(block)
        if not lst:
            self._free_lists.pop(key, None)
            sl_map = self._sl_bitmaps.get(fl, 0) & ~(1 << sl)
            self._sl_bitmaps[fl] = sl_map
            if sl_map == 0:
                self._fl_bitmap &= ~(1 << fl)

    def _coalesce(self, block: _Block) -> None:
        """Try to merge *block* with adjacent free blocks."""
        fl, sl = self._mapping(block.size)
        key = (fl, sl)
        lst = self._free_lists.get(key, [])
        merged = True
        while merged:
            merged = False
            for other in lst:
                if other is block:
                    continue
                if block.offset + block.size == other.offset:
                    self._remove_free(other)
                    self._remove_free(block)
                    block.size += other.size
                    self._insert_free(block)
                    fl, sl = self._mapping(block.size)
                    key = (fl, sl)
                    lst = self._free_lists.get(key, [])
                    merged = True
                    break
                if other.offset + other.size == block.offset:
                    self._remove_free(other)
                    self._remove_free(block)
                    other.size += block.size
                    block = other
                    self._insert_free(block)
                    fl, sl = self._mapping(block.size)
                    key = (fl, sl)
                    lst = self._free_lists.get(key, [])
                    merged = True
                    break

    @property
    def buffer(self) -> bytearray:
        return self._buffer
