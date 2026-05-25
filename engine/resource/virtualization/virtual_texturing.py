"""Virtual texture system with page table management and LRU eviction."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from engine.resource.constants import PAGE_SIZE, PHYSICAL_POOL_TILES


@dataclass(slots=True)
class Page:
    """A virtual texture page mapped to a physical tile."""

    page_x: int
    page_y: int
    mip_level: int
    physical_x: int = -1
    physical_y: int = -1
    is_resident: bool = False


PageKey = tuple[int, int, int]


class PhysicalTexturePool:
    """Fixed-size pool of physical texture tiles with free-list allocation."""

    __slots__ = ("_capacity", "_free", "_next_id")

    def __init__(self, capacity: int = PHYSICAL_POOL_TILES) -> None:
        self._capacity = capacity
        self._free: list[tuple[int, int]] = []
        self._next_id: int = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def allocated(self) -> int:
        return self._next_id - len(self._free)

    def is_full(self) -> bool:
        return not self._free and self._next_id >= self._capacity

    def allocate(self) -> tuple[int, int] | None:
        if self._free:
            return self._free.pop()
        if self._next_id < self._capacity:
            cols = max(1, int(self._capacity ** 0.5))
            px = self._next_id % cols
            py = self._next_id // cols
            self._next_id += 1
            return (px, py)
        return None

    def release(self, px: int, py: int) -> None:
        self._free.append((px, py))


class PageTable:
    """Maps virtual page coordinates to physical locations with LRU tracking."""

    __slots__ = ("_pages", "_lru", "_pool", "_pending_requests")

    def __init__(self, pool: PhysicalTexturePool) -> None:
        self._pages: dict[PageKey, Page] = {}
        self._lru: OrderedDict[PageKey, None] = OrderedDict()
        self._pool = pool
        self._pending_requests: list[PageKey] = []

    def request_page(self, page_x: int, page_y: int, mip_level: int) -> Page:
        """Request a virtual texture page. Allocates physical tile if possible."""
        key: PageKey = (page_x, page_y, mip_level)

        if key in self._pages:
            page = self._pages[key]
            if page.is_resident:
                self._lru.move_to_end(key)
            return page

        page = Page(page_x=page_x, page_y=page_y, mip_level=mip_level)
        self._pages[key] = page
        self._pending_requests.append(key)
        return page

    def _make_resident(self, key: PageKey) -> bool:
        """Attempt to make a page resident by allocating a physical tile."""
        page = self._pages.get(key)
        if page is None or page.is_resident:
            return page is not None and page.is_resident

        if self._pool.is_full():
            if not self._evict_lru():
                return False

        loc = self._pool.allocate()
        if loc is None:
            return False

        page.physical_x, page.physical_y = loc
        page.is_resident = True
        self._lru[key] = None
        return True

    def _evict_lru(self) -> bool:
        """Evict the least recently used resident page."""
        if not self._lru:
            return False
        oldest_key, _ = self._lru.popitem(last=False)
        page = self._pages.get(oldest_key)
        if page and page.is_resident:
            self._pool.release(page.physical_x, page.physical_y)
            page.physical_x = -1
            page.physical_y = -1
            page.is_resident = False
        return True

    def evict_page(self, page_x: int, page_y: int, mip_level: int) -> None:
        """Explicitly evict a page from the physical pool."""
        key: PageKey = (page_x, page_y, mip_level)
        page = self._pages.get(key)
        if page and page.is_resident:
            self._pool.release(page.physical_x, page.physical_y)
            page.physical_x = -1
            page.physical_y = -1
            page.is_resident = False
            self._lru.pop(key, None)

    def get_resident_pages(self) -> list[Page]:
        """Return all currently resident pages."""
        return [p for p in self._pages.values() if p.is_resident]

    def get_feedback(self) -> list[PageKey]:
        """Return list of requested but non-resident page keys."""
        return [k for k in self._pending_requests if not self._pages[k].is_resident]

    def update(self) -> None:
        """Process pending requests, making pages resident where possible."""
        for key in self._pending_requests:
            self._make_resident(key)
        self._pending_requests.clear()


class VirtualTextureSystem:
    """High-level virtual texture system managing page table and physical pool."""

    __slots__ = ("_pool", "_page_table")

    def __init__(self, pool_tiles: int = PHYSICAL_POOL_TILES) -> None:
        self._pool = PhysicalTexturePool(capacity=pool_tiles)
        self._page_table = PageTable(self._pool)

    @property
    def pool(self) -> PhysicalTexturePool:
        return self._pool

    @property
    def page_table(self) -> PageTable:
        return self._page_table

    def request_page(self, page_x: int, page_y: int, mip_level: int) -> Page:
        return self._page_table.request_page(page_x, page_y, mip_level)

    def evict_page(self, page_x: int, page_y: int, mip_level: int) -> None:
        self._page_table.evict_page(page_x, page_y, mip_level)

    def get_resident_pages(self) -> list[Page]:
        return self._page_table.get_resident_pages()

    def get_feedback(self) -> list[PageKey]:
        return self._page_table.get_feedback()

    def update(self) -> None:
        self._page_table.update()
