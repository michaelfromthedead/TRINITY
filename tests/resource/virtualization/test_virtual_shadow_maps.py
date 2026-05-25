"""Tests for the virtual shadow map system."""

import pytest

from engine.resource.virtualization.virtual_shadow_maps import (
    NUM_CLIPMAP_LEVELS,
    SHADOW_PAGE_SIZE,
    ShadowClipmapLevel,
    ShadowPage,
    VirtualShadowMapSystem,
)


class TestShadowPage:
    def test_defaults(self) -> None:
        page = ShadowPage(page_x=0, page_y=0, clipmap_level=0)
        assert page.is_cached is False
        assert page.is_dirty is True


class TestVirtualShadowMapSystem:
    def test_init_creates_clipmap_levels(self) -> None:
        vsm = VirtualShadowMapSystem()
        assert len(vsm.levels) == NUM_CLIPMAP_LEVELS
        for i, lvl in enumerate(vsm.levels):
            assert lvl.level == i
            assert lvl.world_size > 0

    def test_update_light_creates_dirty_pages(self) -> None:
        vsm = VirtualShadowMapSystem()
        vsm.update_light(light_dir=(0.0, -1.0, 0.0), camera_pos=(0.0, 0.0, 0.0))
        dirty = vsm.get_dirty_pages()
        # All pages should be dirty after initial light update
        stats = vsm.get_cache_stats()
        assert len(dirty) == stats["total_pages"]
        assert all(p.is_dirty for p in dirty)

    def test_mark_rendered_clears_dirty(self) -> None:
        vsm = VirtualShadowMapSystem()
        vsm.update_light(light_dir=(0.0, -1.0, 0.0), camera_pos=(0.0, 0.0, 0.0))
        dirty = vsm.get_dirty_pages()
        assert len(dirty) > 0
        page = dirty[0]
        vsm.mark_rendered(page.page_x, page.page_y, page.clipmap_level)
        assert page.is_dirty is False
        assert page.is_cached is True

    def test_invalidate_region_marks_dirty(self) -> None:
        vsm = VirtualShadowMapSystem()
        vsm.update_light(light_dir=(0.0, -1.0, 0.0), camera_pos=(0.0, 0.0, 0.0))
        # Render all pages
        for page in vsm.get_dirty_pages():
            vsm.mark_rendered(page.page_x, page.page_y, page.clipmap_level)
        assert len(vsm.get_dirty_pages()) == 0

        # Invalidate a region
        total_pages = vsm.get_cache_stats()["total_pages"]
        vsm.invalidate_region(x=0.0, y=0.0, radius=10.0)
        dirty_after = vsm.get_dirty_pages()
        # Invalidation should dirty some but not necessarily all pages
        assert 0 < len(dirty_after) <= total_pages

    def test_cache_stats(self) -> None:
        vsm = VirtualShadowMapSystem()
        vsm.update_light(light_dir=(0.0, -1.0, 0.0), camera_pos=(0.0, 0.0, 0.0))
        stats = vsm.get_cache_stats()
        assert "total_pages" in stats
        assert "cached_pages" in stats
        assert "dirty_pages" in stats
        assert "cache_hit_rate" in stats
        assert "num_levels" in stats
        assert stats["num_levels"] == NUM_CLIPMAP_LEVELS
        assert stats["total_pages"] > 0
        assert stats["cached_pages"] == 0
        assert stats["cache_hit_rate"] == 0.0
        assert stats["dirty_pages"] == stats["total_pages"]

    def test_cache_stats_after_rendering(self) -> None:
        vsm = VirtualShadowMapSystem()
        vsm.update_light(light_dir=(0.0, -1.0, 0.0), camera_pos=(0.0, 0.0, 0.0))
        for page in vsm.get_dirty_pages():
            vsm.mark_rendered(page.page_x, page.page_y, page.clipmap_level)
        stats = vsm.get_cache_stats()
        assert stats["dirty_pages"] == 0
        assert stats["cached_pages"] == stats["total_pages"]
        assert stats["cache_hit_rate"] == pytest.approx(1.0)

    def test_constants(self) -> None:
        assert NUM_CLIPMAP_LEVELS > 0
        assert SHADOW_PAGE_SIZE > 0 and SHADOW_PAGE_SIZE % 2 == 0
