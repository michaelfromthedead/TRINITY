"""
White-box tests for channel edge cases.

Tests all channel types (Unreliable, Reliable, ReliableOrdered, Sequenced)
and the ChannelManager for edge case behavior including sequence wraparound,
retransmission, ordering, and duplicate handling.
"""

from __future__ import annotations

import time
from unittest import mock
import pytest

from engine.networking.transport.channel import (
    ChannelType, ChannelConfig, ChannelStats,
    UnreliableChannel, ReliableChannel, ReliableOrderedChannel,
    SequencedChannel, ChannelManager, PendingPacket,
)
from engine.networking.transport.packet import Packet, PacketType, PacketFlags
from engine.networking.config import DEFAULT_CONFIG


class TestChannelType:
    """ChannelType enum tests."""

    def test_has_correct_values(self):
        assert ChannelType.UNRELIABLE == 0
        assert ChannelType.RELIABLE_UNORDERED == 1
        assert ChannelType.RELIABLE_ORDERED == 2
        assert ChannelType.SEQUENCED == 3


class TestChannelConfig:
    """ChannelConfig defaults test."""

    def test_defaults_from_config(self):
        cfg = ChannelConfig(ChannelType.UNRELIABLE)
        assert cfg.max_pending == DEFAULT_CONFIG.CHANNEL_MAX_PENDING
        assert cfg.initial_rtt == DEFAULT_CONFIG.CHANNEL_INITIAL_RTT
        assert cfg.max_retries == DEFAULT_CONFIG.CHANNEL_MAX_RETRIES
        assert cfg.ack_timeout == DEFAULT_CONFIG.CHANNEL_ACK_TIMEOUT
        assert cfg.ordering_buffer_size == DEFAULT_CONFIG.CHANNEL_ORDERING_BUFFER_SIZE


class TestUnreliableChannel:
    """UnreliableChannel edge case tests."""

    def test_channel_type(self):
        channel = UnreliableChannel(0)
        assert channel.channel_type == ChannelType.UNRELIABLE

    def test_channel_id(self):
        channel = UnreliableChannel(5)
        assert channel.channel_id == 5

    def test_send_returns_data_packet(self):
        channel = UnreliableChannel(0)
        packets = channel.send(b'hello')
        assert len(packets) == 1
        assert packets[0].header.packet_type == PacketType.DATA
        assert packets[0].payload == b'hello'

    def test_send_empty_data(self):
        channel = UnreliableChannel(0)
        packets = channel.send(b'')
        assert len(packets) == 1
        assert packets[0].payload == b''

    def test_receive_returns_payload(self):
        channel = UnreliableChannel(0)
        packet = Packet.create(PacketType.DATA, b'hello', sequence=1)
        result = channel.receive(packet)
        assert result == b'hello'

    def test_receive_updates_stats(self):
        channel = UnreliableChannel(0)
        packet = Packet.create(PacketType.DATA, b'hello', sequence=1)
        channel.receive(packet)
        assert channel._stats.packets_received == 1

    def test_process_ack_returns_empty(self):
        channel = UnreliableChannel(0)
        assert channel.process_ack(0, 0) == []

    def test_update_returns_empty(self):
        channel = UnreliableChannel(0)
        assert channel.update(0.016) == []

    def test_get_next_sequence_increments(self):
        channel = UnreliableChannel(0)
        s1 = channel.get_next_sequence()
        s2 = channel.get_next_sequence()
        assert s2 == (s1 + 1) & 0xFFFF

    def test_stats_property(self):
        channel = UnreliableChannel(0)
        assert isinstance(channel.stats, ChannelStats)


class TestReliableChannel:
    """ReliableChannel edge case tests."""

    def test_channel_type(self):
        channel = ReliableChannel(1)
        assert channel.channel_type == ChannelType.RELIABLE_UNORDERED

    def test_send_returns_packets_with_reliable_flag(self):
        channel = ReliableChannel(1)
        packets = channel.send(b'hello')
        assert len(packets) >= 1
        for p in packets:
            assert p.header.has_flag(PacketFlags.RELIABLE)

    def test_send_tracks_pending_acks(self):
        channel = ReliableChannel(1)
        channel.send(b'hello')
        assert len(channel._pending) == 1

    def test_receive_returns_payload(self):
        channel = ReliableChannel(1)
        packet = Packet.create(PacketType.RELIABLE_DATA, b'data', sequence=1, flags=PacketFlags.RELIABLE)
        result = channel.receive(packet)
        assert result == b'data'

    def test_receive_duplicate_returns_none(self):
        channel = ReliableChannel(1)
        packet = Packet.create(PacketType.RELIABLE_DATA, b'data', sequence=1, flags=PacketFlags.RELIABLE)
        channel.receive(packet)
        result = channel.receive(packet)
        assert result is None

    def test_process_ack_removes_pending(self):
        channel = ReliableChannel(1)
        with mock.patch('time.time', return_value=100.0):
            channel.send(b'data')
            assert len(channel._pending) == 1
            seq = list(channel._pending.keys())[0]
            channel.process_ack(seq, 0)
            assert len(channel._pending) == 0

    def test_update_retransmits_unacked(self):
        channel = ReliableChannel(1)
        with mock.patch('time.time', return_value=100.0):
            channel.send(b'data')
            # Set retransmit_time in the past
            for seq, pp in channel._pending.items():
                pp.retransmit_time = 50.0
        with mock.patch('time.time', return_value=200.0):
            packets = channel.update(0.016)
            assert len(packets) == 1
            assert packets[0].header.sequence in channel._pending

    def test_update_gives_up_after_max_retries(self):
        cfg = ChannelConfig(ChannelType.RELIABLE_UNORDERED, max_retries=2)
        channel = ReliableChannel(1, cfg)
        with mock.patch('time.time', return_value=100.0):
            channel.send(b'data')
        seq = list(channel._pending.keys())[0]
        while seq in channel._pending:
            channel._pending[seq].retransmit_time = 0.0
            channel.update(1.0)
        assert seq not in channel._pending

    def test_get_ack_data_returns_tuple(self):
        channel = ReliableChannel(1)
        ack, bits = channel.get_ack_data()
        assert isinstance(ack, int)
        assert isinstance(bits, int)


class TestReliableOrderedChannel:
    """ReliableOrderedChannel edge case tests."""

    def test_in_order_delivers(self):
        channel = ReliableOrderedChannel(2)
        p1 = Packet.create(PacketType.RELIABLE_DATA, b'first', sequence=0, flags=PacketFlags.RELIABLE | PacketFlags.ORDERED)
        result = channel.receive(p1)
        assert result == b'first'

    def test_out_of_order_buffers(self):
        channel = ReliableOrderedChannel(2)
        p2 = Packet.create(PacketType.RELIABLE_DATA, b'second', sequence=1, flags=PacketFlags.RELIABLE | PacketFlags.ORDERED)
        result = channel.receive(p2)
        assert result is None
        assert channel.get_buffered_count() == 1

    def test_gap_filled_delivers_buffered(self):
        channel = ReliableOrderedChannel(2)
        p2 = Packet.create(PacketType.RELIABLE_DATA, b'second', sequence=1, flags=PacketFlags.RELIABLE | PacketFlags.ORDERED)
        channel.receive(p2)
        p1 = Packet.create(PacketType.RELIABLE_DATA, b'first', sequence=0, flags=PacketFlags.RELIABLE | PacketFlags.ORDERED)
        result = channel.receive(p1)
        assert result == b'firstsecond'

    def test_duplicate_returns_none(self):
        channel = ReliableOrderedChannel(2)
        p = Packet.create(PacketType.RELIABLE_DATA, b'data', sequence=0, flags=PacketFlags.RELIABLE | PacketFlags.ORDERED)
        channel.receive(p)
        result = channel.receive(p)
        assert result is None

    def test_get_buffered_count(self):
        channel = ReliableOrderedChannel(2)
        assert channel.get_buffered_count() == 0
        p = Packet.create(PacketType.RELIABLE_DATA, b'data', sequence=5, flags=PacketFlags.RELIABLE | PacketFlags.ORDERED)
        channel.receive(p)
        assert channel.get_buffered_count() == 1

    def test_sequence_wraparound(self):
        channel = ReliableOrderedChannel(2)
        channel._next_deliver_sequence = 0xFFFF
        p_ffff = Packet.create(PacketType.RELIABLE_DATA, b'wrap', sequence=0xFFFF, flags=PacketFlags.RELIABLE | PacketFlags.ORDERED)
        result = channel.receive(p_ffff)
        assert result == b'wrap'
        p_0000 = Packet.create(PacketType.RELIABLE_DATA, b'zero', sequence=0, flags=PacketFlags.RELIABLE | PacketFlags.ORDERED)
        result = channel.receive(p_0000)
        assert result == b'zero'


class TestSequencedChannel:
    """SequencedChannel edge case tests."""

    def test_send_creates_sequenced_data(self):
        channel = SequencedChannel(3)
        packets = channel.send(b'position')
        assert len(packets) == 1
        assert packets[0].header.packet_type == PacketType.SEQUENCED_DATA

    def test_receive_newer_returns_payload(self):
        channel = SequencedChannel(3)
        p1 = Packet.create(PacketType.SEQUENCED_DATA, b'old', sequence=1)
        result = channel.receive(p1)
        assert result == b'old'

    def test_receive_older_dropped(self):
        channel = SequencedChannel(3)
        p1 = Packet.create(PacketType.SEQUENCED_DATA, b'new', sequence=5)
        channel.receive(p1)
        p2 = Packet.create(PacketType.SEQUENCED_DATA, b'old', sequence=3)
        result = channel.receive(p2)
        assert result is None

    def test_receive_same_sequence_dropped(self):
        channel = SequencedChannel(3)
        p1 = Packet.create(PacketType.SEQUENCED_DATA, b'first', sequence=5)
        channel.receive(p1)
        p2 = Packet.create(PacketType.SEQUENCED_DATA, b'second', sequence=5)
        result = channel.receive(p2)
        assert result is None

    def test_first_packet_always_accepted(self):
        channel = SequencedChannel(3)
        p = Packet.create(PacketType.SEQUENCED_DATA, b'first', sequence=100)
        result = channel.receive(p)
        assert result == b'first'

    def test_process_ack_returns_empty(self):
        channel = SequencedChannel(3)
        assert channel.process_ack(0, 0) == []

    def test_update_returns_empty(self):
        channel = SequencedChannel(3)
        assert channel.update(0.016) == []


class TestChannelManager:
    """ChannelManager edge case tests."""

    def test_create_all_types(self):
        mgr = ChannelManager()
        for ct in ChannelType:
            ch = mgr.create_channel(int(ct), ct)
            assert ch is not None

    def test_get_channel_by_id(self):
        mgr = ChannelManager()
        ch = mgr.create_channel(0, ChannelType.UNRELIABLE)
        assert mgr.get_channel(0) is ch

    def test_get_channel_by_type(self):
        mgr = ChannelManager()
        ch = mgr.create_channel(0, ChannelType.UNRELIABLE)
        assert mgr.get_channel_by_type(ChannelType.UNRELIABLE) is ch

    def test_remove_channel_removes_id_and_type(self):
        mgr = ChannelManager()
        mgr.create_channel(0, ChannelType.UNRELIABLE)
        mgr.remove_channel(0)
        assert mgr.get_channel(0) is None
        assert mgr.get_channel_by_type(ChannelType.UNRELIABLE) is None

    def test_remove_unknown_channel_does_nothing(self):
        mgr = ChannelManager()
        mgr.remove_channel(99)
        assert mgr.get_channel(99) is None

    def test_update_returns_packets(self):
        mgr = ChannelManager()
        mgr.create_channel(0, ChannelType.UNRELIABLE)
        mgr.create_channel(1, ChannelType.RELIABLE_UNORDERED)
        result = mgr.update(0.016)
        assert isinstance(result, list)

    def test_get_aggregate_stats_empty(self):
        mgr = ChannelManager()
        stats = mgr.get_aggregate_stats()
        assert stats.packets_sent == 0

    def test_get_aggregate_stats_with_channels(self):
        mgr = ChannelManager()
        ch = mgr.create_channel(0, ChannelType.UNRELIABLE)
        ch.send(b'hello')
        stats = mgr.get_aggregate_stats()
        assert stats.packets_sent >= 1
        assert stats.bytes_sent >= 5

    def test_custom_config(self):
        cfg = ChannelConfig(ChannelType.UNRELIABLE, max_pending=50)
        mgr = ChannelManager()
        ch = mgr.create_channel(0, ChannelType.UNRELIABLE, cfg)
        assert ch._config.max_pending == 50


class TestChannelStats:
    """ChannelStats defaults test."""

    def test_default_values(self):
        stats = ChannelStats()
        assert stats.packets_sent == 0
        assert stats.packets_received == 0
        assert stats.packets_lost == 0
        assert stats.packets_retransmitted == 0
        assert stats.bytes_sent == 0
        assert stats.bytes_received == 0
        assert stats.average_rtt == 0.0
        assert stats.pending_acks == 0
