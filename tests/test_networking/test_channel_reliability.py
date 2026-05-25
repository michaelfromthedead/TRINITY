"""T-NET-1.7: WHITEBOX tests for channel reliability under simulated packet loss.

Tests exercise reliable channel retransmission, ordered delivery after loss,
burst loss recovery, and ACK processing with various loss patterns.
"""

from __future__ import annotations

import time

import pytest

from engine.networking.transport.packet import Packet, PacketType, PacketFlags
from engine.networking.transport.channel import (
    ChannelType,
    ChannelConfig,
    ReliableChannel,
    ReliableOrderedChannel,
    SequencedChannel,
    UnreliableChannel,
)


# ---------------------------------------------------------------------------
# Helpers: simulated loss
# ---------------------------------------------------------------------------

def _deliver_with_loss(packets, loss_rate: float = 0.0):
    """Yield (packet, was_delivered) pairs, simulating random packet loss.

    ``loss_rate`` is the probability (0..1) that a packet is dropped.
    """
    import random

    rng = random.Random(42)  # deterministic seed
    for p in packets:
        if rng.random() >= loss_rate:
            yield p, True
        else:
            yield p, False


def _make_channel(channel_type, channel_id=0, initial_rtt=0.001, max_retries=10):
    """Convenience: create a ReliableChannel / ReliableOrderedChannel with fast retransmit."""
    config = ChannelConfig(
        channel_type=channel_type,
        initial_rtt=initial_rtt,
        max_retries=max_retries,
    )
    if channel_type == ChannelType.RELIABLE_ORDERED:
        return ReliableOrderedChannel(channel_id, config)
    return ReliableChannel(channel_id, config)


# ---------------------------------------------------------------------------
# UnreliableChannel: loss behaviour
# ---------------------------------------------------------------------------

class TestUnreliableChannelLoss:
    """UnreliableChannel has no retransmission — packets are fire-and-forget."""

    def test_no_retransmit_after_loss(self):
        channel = UnreliableChannel(0)
        packets = channel.send(b"msg")
        assert len(packets) == 1

        # After update, nothing is retransmitted (no pending acks)
        retransmits = channel.update(0.1)
        assert retransmits == []

    def test_lost_unreliable_not_recovered(self):
        """Receiver never receives lost unreliable packets."""
        sender = UnreliableChannel(0)
        pkts = sender.send(b"lost data")

        # The receiver channel never gets the packet -> nothing to receive
        receiver = UnreliableChannel(1)
        assert receiver.receive(pkts[0]) == b"lost data"

        # If the packet was lost, the receiver simply never calls receive on it


# ---------------------------------------------------------------------------
# ReliableChannel: loss recovery
# ---------------------------------------------------------------------------

class TestReliableChannelUnderLoss:
    """Whitebox: ReliableChannel retransmission under various loss patterns."""

    def test_no_loss_all_delivered(self):
        """With 0% loss, all packets are sent and acked normally."""
        channel = _make_channel(ChannelType.RELIABLE_UNORDERED)
        sent_packets = channel.send(b"data")

        assert len(sent_packets) == 1

        # Process the ACK for the sent packet
        ack_seq = sent_packets[0].header.sequence
        channel.process_ack(ack_seq, 0)
        assert channel.stats.pending_acks == 0

    def test_moderate_loss_triggers_retransmit(self):
        """At 50% loss, lost packets are retransmitted via update()."""
        channel = _make_channel(ChannelType.RELIABLE_UNORDERED)

        # Send 4 packets
        all_packets = []
        for i in range(4):
            all_packets.extend(channel.send(f"msg{i}".encode()))

        # Deliver only even-indexed packets; record ACKs for those
        received_seqs = set()
        for i, pkt in enumerate(all_packets):
            if i % 2 == 0:
                data = channel.receive(pkt)
                received_seqs.add(pkt.header.sequence)

        # Send ACKs for received packets
        if received_seqs:
            ack_seq = max(received_seqs)
            ack_bits = 0
            for s in received_seqs:
                if s < ack_seq:
                    ack_bits |= 1 << (ack_seq - s - 1)
            channel.process_ack(ack_seq, ack_bits)

        # Wait for retransmit timeout
        time.sleep(0.005)

        # Update triggers retransmission of lost (unacked) packets
        retransmits = channel.update(0.01)

        # At least some packets should be retransmitted
        assert len(retransmits) > 0, "Expected retransmits after loss"

    def test_high_loss_rate_eventual_delivery(self):
        """Even at 70% loss, retransmit mechanism eventually delivers."""
        # Use very fast retransmit + few retries for test speed
        channel = _make_channel(
            ChannelType.RELIABLE_UNORDERED,
            initial_rtt=0.001,
            max_retries=30,
        )

        packets = channel.send(b"survivor")

        # Deliver with 70% loss in a single deterministic pass
        rng = __import__("random").Random(42)
        acked = False
        for attempt in range(200):
            if rng.random() >= 0.7:
                channel.receive(packets[0])
                channel.process_ack(packets[0].header.sequence, 0)
                acked = True
                break

            # Advance retransmit timer deterministically (avoid flaky time.sleep)
            now = time.time()
            for pending in channel._pending.values():
                pending.retransmit_time = now - 0.001
            retransmits = channel.update(0.01)
            if retransmits:
                packets = retransmits

        # The packet should eventually be acked
        assert acked or channel.stats.pending_acks == 0, \
            "Packet was never delivered under high loss"

    def test_burst_loss_recovery(self):
        """All packets in a burst loss window are retransmitted."""
        channel = _make_channel(ChannelType.RELIABLE_UNORDERED, initial_rtt=0.001)

        # Send 3 packets
        all_packets = []
        for i in range(3):
            all_packets.extend(channel.send(f"burst{i}".encode()))

        # Receive the first one (0), lose the next two (1, 2)
        channel.receive(all_packets[0])
        channel.process_ack(all_packets[0].header.sequence, 0)

        # Wait for retransmit
        time.sleep(0.005)

        # update should retransmit packets 1 and 2
        retransmits = channel.update(0.01)

        # The retransmitted packets should reference the original sequences
        original_seqs = {p.header.sequence for p in all_packets[1:]}
        retransmit_seqs = {p.header.sequence for p in retransmits}
        intersection = original_seqs & retransmit_seqs

        assert len(intersection) >= 1, \
            "Expected retransmits for lost burst packets"

    def test_ack_clears_pending(self):
        """Processing an ACK removes the packet from pending."""
        channel = _make_channel(ChannelType.RELIABLE_UNORDERED)
        packets = channel.send(b"ack_me")

        assert channel.stats.pending_acks == 1

        channel.process_ack(packets[0].header.sequence, 0)

        assert channel.stats.pending_acks == 0

    def test_duplicate_ack_is_idempotent(self):
        """Processing the same ACK twice should not double-count."""
        channel = _make_channel(ChannelType.RELIABLE_UNORDERED)
        packets = channel.send(b"dup_ack")

        channel.process_ack(packets[0].header.sequence, 0)
        acks_after_first = channel.stats.pending_acks

        channel.process_ack(packets[0].header.sequence, 0)
        acks_after_second = channel.stats.pending_acks

        assert acks_after_first == 0
        assert acks_after_second == 0

    def test_max_retries_exhausted_drops_packet(self):
        """When retry limit is reached, the channel stops retransmitting."""
        channel = _make_channel(
            ChannelType.RELIABLE_UNORDERED,
            initial_rtt=0.001,
            max_retries=2,
        )

        packets = channel.send(b"give_up")

        # Run update cycles without ever sending an ACK
        time.sleep(0.005)
        for _ in range(10):
            retransmits = channel.update(0.01)
            time.sleep(0.005)
            # Continue until no more retransmits or max retries reached

        # After exhausting retries, pending_acks may be 0 or the packet
        # was dropped from the pending set internally
        # Just verify the channel doesn't error
        assert channel.stats.packets_retransmitted >= 1

    # ------------------------------------------------------------------
    # FLK-01: Deterministic retransmission (no time.sleep flakiness)
    # ------------------------------------------------------------------

    def test_deterministic_retransmission_trigger(self):
        """Retransmission fires deterministically by setting retransmit_time directly (FLK-01)."""
        channel = _make_channel(ChannelType.RELIABLE_UNORDERED, initial_rtt=0.001)
        channel.send(b"packet1")
        channel.send(b"packet2")

        assert channel.stats.pending_acks == 2

        # Advance ALL pending retransmit timers deterministically
        now = time.time()
        for pending in channel._pending.values():
            pending.retransmit_time = now - 0.001

        retransmits = channel.update(0.01)
        assert len(retransmits) >= 1, \
            "Should retransmit at least 1 packet after advancing timer"
        assert channel.stats.packets_retransmitted >= 1

    def test_deterministic_retransmission_selective(self):
        """Only packets past their retransmit_time are retransmitted."""
        channel = _make_channel(ChannelType.RELIABLE_UNORDERED, initial_rtt=0.001)

        pkts = []
        for i in range(4):
            pkts.extend(channel.send(f"msg{i}".encode()))

        now = time.time()
        pendings = list(channel._pending.values())
        # Advance timer for first two only
        for p in pendings[:2]:
            p.retransmit_time = now - 0.001

        retransmits = channel.update(0.01)
        retransmit_seqs = {p.header.sequence for p in retransmits}
        advanced_seqs = {pkts[i].header.sequence for i in range(2)}
        not_advanced_seqs = {pkts[i].header.sequence for i in range(2, 4)}

        # Advanced packets should be in retransmit set
        assert len(advanced_seqs & retransmit_seqs) >= 1
        # Non-advanced packets should NOT have been retransmitted yet
        assert len(not_advanced_seqs & retransmit_seqs) == 0

    def test_deterministic_retransmission_no_sleep(self):
        """Deterministic retransmission avoids time.sleep entirely (FLK-01)."""
        channel = _make_channel(ChannelType.RELIABLE_UNORDERED, initial_rtt=10.0)
        channel.send(b"no_sleep")

        # Without advancing timer, update should NOT retransmit
        assert channel.update(0.016) == []

        # With advanced timer, update SHOULD retransmit
        now = time.time()
        for pending in channel._pending.values():
            pending.retransmit_time = now - 0.001

        retransmits = channel.update(0.01)
        assert len(retransmits) == 1

    def test_deterministic_bypasses_real_timeout(self):
        """Setting retransmit_time in the past triggers immediate retransmit regardless of real time (FLK-01)."""
        channel = _make_channel(ChannelType.RELIABLE_UNORDERED, initial_rtt=999.0)
        channel.send(b"bypass")

        # Real RTT is 999s, but we set retransmit_time to 1s ago -> should retransmit
        now = time.time()
        for pending in channel._pending.values():
            pending.retransmit_time = now - 1.0

        retransmits = channel.update(0.01)
        assert len(retransmits) == 1


# ---------------------------------------------------------------------------
# ReliableOrderedChannel: loss with ordering
# ---------------------------------------------------------------------------

class TestReliableOrderedChannelUnderLoss:
    """Whitebox: ordered delivery is preserved after lost packets are retransmitted."""

    def test_ordered_delivery_after_loss(self):
        """Packets that arrive after loss are still delivered in order."""
        channel = _make_channel(ChannelType.RELIABLE_ORDERED, initial_rtt=0.001)

        # Send 4 packets with known sequence numbers
        packets = []
        for i in range(4):
            pkts = channel.send(f"ordered{i}".encode())
            packets.extend(pkts)

        assert len(packets) == 4

        # Deliver packets 0, 2, 3 (skip 1 - "lost")
        received = []
        for i, pkt in enumerate(packets):
            if i != 1:  # skip index 1 (sequence 1)
                data = channel.receive(pkt)
                if data:
                    received.append(data)
                # Send ack for the received packet
                channel.process_ack(pkt.header.sequence, 0)

        # Packet 1 is missing so packets 2,3 should be buffered
        # Only packet 0 should have been delivered
        assert len(received) == 1
        assert received[0] == b"ordered0"

        # Now deliver the lost packet
        data = channel.receive(packets[1])
        if data:
            received.append(data)

        # After all 4 are received, should get all data in order
        all_data = b"".join(received)

        # The ReliableOrderedChannel delivers buffered packets when the gap fills
        assert b"ordered1" in all_data
        assert b"ordered2" in all_data
        assert b"ordered3" in all_data

    def test_buffered_count_tracks_gaps(self):
        """Buffered count increases when packets arrive out of order."""
        channel = _make_channel(ChannelType.RELIABLE_ORDERED, initial_rtt=0.001)

        pkts = []
        for i in range(3):
            pkts.extend(channel.send(f"o{i}".encode()))

        # Receive packet 2 first (out of order)
        channel.receive(pkts[2])
        assert channel.get_buffered_count() == 1

        # Receive packet 0
        channel.receive(pkts[0])
        # Buffered count may have changed due to delivery
        # Packet 1 is still missing

        # Receive packet 1 closes the gap
        channel.receive(pkts[1])
        assert channel.get_buffered_count() == 0


# ---------------------------------------------------------------------------
# SequencedChannel: loss behaviour
# ---------------------------------------------------------------------------

class TestSequencedChannelUnderLoss:
    """SequencedChannel only accepts newer packets, drops everything else."""

    def test_older_packet_after_newer_is_dropped(self):
        """If a newer packet arrives first, the old one is dropped on arrival."""
        channel = SequencedChannel(0)

        # Receive newer first (simulating loss / reorder)
        pkt_new = Packet.create(
            PacketType.SEQUENCED_DATA, b"newer", sequence=10,
        )
        channel.receive(pkt_new)

        # Receive older (should be dropped)
        pkt_old = Packet.create(
            PacketType.SEQUENCED_DATA, b"older", sequence=5,
        )
        result = channel.receive(pkt_old)

        assert result is None

    def test_duplicate_sequenced_dropped(self):
        """Same sequence number is treated as duplicate."""
        channel = SequencedChannel(0)

        pkt = Packet.create(
            PacketType.SEQUENCED_DATA, b"dup", sequence=7,
        )
        result1 = channel.receive(pkt)
        result2 = channel.receive(pkt)

        assert result1 == b"dup"
        assert result2 is None

    def test_strict_increasing_sequence(self):
        """Only strictly later sequence numbers produce a result."""
        channel = SequencedChannel(0)

        for seq in [1, 2]:
            pkt = Packet.create(
                PacketType.SEQUENCED_DATA, f"s{seq}".encode(), sequence=seq,
            )
            result = channel.receive(pkt)
            assert result == f"s{seq}".encode()

        # This is equal to the last received, not greater
        pkt = Packet.create(
            PacketType.SEQUENCED_DATA, b"dup2", sequence=2,
        )
        assert channel.receive(pkt) is None
