"""Tests for TextureStreamManager."""

from engine.resource.streaming.texture_streaming import MipStreamRequest, TextureStreamManager


class TestMipStreamRequest:
    def test_priority_equals_target_mip(self) -> None:
        req = MipStreamRequest(texture_id="t", target_mip_level=3, current_mip_level=5)
        assert req.priority == 3

    def test_lower_mip_is_higher_priority(self) -> None:
        low = MipStreamRequest(texture_id="t", target_mip_level=0)
        high = MipStreamRequest(texture_id="t", target_mip_level=5)
        assert low.priority < high.priority


class TestTextureStreamManager:
    def test_unloaded_returns_negative_one(self) -> None:
        mgr = TextureStreamManager()
        assert mgr.get_resident_mip("unknown") == -1

    def test_request_and_update_loads_mip(self) -> None:
        mgr = TextureStreamManager()
        mgr.request_mip("tex_01", mip_level=2)
        mgr.update()
        assert mgr.get_resident_mip("tex_01") == 2

    def test_multiple_mip_requests_best_mip_wins(self) -> None:
        mgr = TextureStreamManager()
        mgr.request_mip("tex_01", mip_level=4)
        mgr.request_mip("tex_01", mip_level=1)
        mgr.update()
        # Lower mip = higher priority, processed last so it wins
        assert mgr.get_resident_mip("tex_01") == 1

    def test_request_records_current_mip(self) -> None:
        mgr = TextureStreamManager()
        mgr.request_mip("tex_01", mip_level=3)
        mgr.update()
        req = mgr.request_mip("tex_01", mip_level=1)
        assert req.current_mip_level == 3
