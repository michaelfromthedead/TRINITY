"""Tests for CheckpointManager."""

import pytest

from engine.core.session.checkpoint import CheckpointManager
from engine.core.session.session import SessionData


class TestCheckpointManager:
    def test_create_and_restore(self):
        mgr = CheckpointManager()
        sd = SessionData(frame_count=10)
        cid = mgr.create_checkpoint(sd)
        restored = mgr.restore_checkpoint(cid)
        assert restored is not None
        assert restored.frame_count == 10

    def test_list_checkpoints(self):
        mgr = CheckpointManager()
        ids = [mgr.create_checkpoint(SessionData(frame_count=i)) for i in range(3)]
        assert mgr.list_checkpoints() == ids

    def test_auto_prune(self):
        mgr = CheckpointManager(max_checkpoints=3)
        ids = [mgr.create_checkpoint(SessionData(frame_count=i)) for i in range(5)]
        listed = mgr.list_checkpoints()
        assert len(listed) == 3
        # oldest two should be pruned
        assert mgr.restore_checkpoint(ids[0]) is None
        assert mgr.restore_checkpoint(ids[1]) is None
        assert mgr.restore_checkpoint(ids[4]) is not None

    def test_restore_nonexistent(self):
        mgr = CheckpointManager()
        assert mgr.restore_checkpoint("nonexistent") is None
