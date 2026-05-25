"""Tests for the virtual texturing system."""

import pytest

from engine.resource.virtualization.virtual_texturing import (
    PAGE_SIZE,
    PHYSICAL_POOL_TILES,
    Page,
    PhysicalTexturePool,
    VirtualTextureSystem,
)


class TestPage:
    def test_default_non_resident(self) -> None:
        page = Page(page_x=0, page_y=0, mip_level=0)
        assert page.is_resident is False
        assert page.physical_x == -1
        assert page.physical_y == -1


class TestPhysicalTexturePool:
    def test_allocate_returns_coordinates(self) -> None:
        pool = PhysicalTexturePool(capacity=4)
        loc = pool.allocate()
        assert loc is not None
        assert isinstance(loc, tuple)
        assert len(loc) == 2
        assert loc == (0, 0)

    def test_full_pool_returns_none(self) -> None:
        pool = PhysicalTexturePool(capacity=2)
        pool.allocate()
        pool.allocate()
        assert pool.is_full()
        assert pool.allocate() is None

    def test_release_allows_realloc(self) -> None:
        pool = PhysicalTexturePool(capacity=1)
        loc = pool.allocate()
        assert pool.is_full()
        pool.release(loc[0], loc[1])
        assert not pool.is_full()
        new_loc = pool.allocate()
        assert new_loc == loc


class TestVirtualTextureSystem:
    def test_request_page_non_resident_before_update(self) -> None:
        vts = VirtualTextureSystem(pool_tiles=8)
        page = vts.request_page(0, 0, 0)
        assert page.is_resident is False

    def test_request_page_resident_after_update(self) -> None:
        vts = VirtualTextureSystem(pool_tiles=8)
        vts.request_page(1, 2, 0)
        vts.update()
        residents = vts.get_resident_pages()
        assert len(residents) == 1
        assert residents[0].page_x == 1
        assert residents[0].page_y == 2
        assert residents[0].is_resident is True

    def test_evict_page(self) -> None:
        vts = VirtualTextureSystem(pool_tiles=8)
        vts.request_page(0, 0, 0)
        vts.update()
        assert len(vts.get_resident_pages()) == 1
        vts.evict_page(0, 0, 0)
        assert len(vts.get_resident_pages()) == 0

    def test_feedback_lists_non_resident(self) -> None:
        vts = VirtualTextureSystem(pool_tiles=8)
        vts.request_page(5, 5, 2)
        feedback = vts.get_feedback()
        assert len(feedback) == 1
        assert feedback[0] == (5, 5, 2)

    def test_feedback_empty_after_update(self) -> None:
        vts = VirtualTextureSystem(pool_tiles=8)
        vts.request_page(5, 5, 2)
        vts.update()
        # After update, pending cleared; new feedback call returns empty
        feedback = vts.get_feedback()
        assert len(feedback) == 0

    def test_lru_eviction_when_pool_full(self) -> None:
        vts = VirtualTextureSystem(pool_tiles=2)
        vts.request_page(0, 0, 0)
        vts.request_page(1, 0, 0)
        vts.update()
        assert len(vts.get_resident_pages()) == 2

        # Request a third page; LRU (0,0,0) should be evicted
        vts.request_page(2, 0, 0)
        vts.update()
        resident_keys = {(p.page_x, p.page_y, p.mip_level) for p in vts.get_resident_pages()}
        assert (2, 0, 0) in resident_keys
        assert len(resident_keys) == 2
        # The oldest (0,0,0) was evicted
        assert (0, 0, 0) not in resident_keys

    def test_lru_touch_prevents_eviction(self) -> None:
        vts = VirtualTextureSystem(pool_tiles=2)
        vts.request_page(0, 0, 0)
        vts.request_page(1, 0, 0)
        vts.update()

        # Touch page (0,0,0) by re-requesting
        vts.request_page(0, 0, 0)

        # Now request a third; LRU is now (1,0,0)
        vts.request_page(2, 0, 0)
        vts.update()
        resident_keys = {(p.page_x, p.page_y, p.mip_level) for p in vts.get_resident_pages()}
        assert (0, 0, 0) in resident_keys
        assert (1, 0, 0) not in resident_keys
        assert (2, 0, 0) in resident_keys

    def test_constants(self) -> None:
        assert PAGE_SIZE > 0 and PAGE_SIZE % 2 == 0
        assert PHYSICAL_POOL_TILES > 0 and PHYSICAL_POOL_TILES >= PAGE_SIZE
