"""
Tests for the TRINITY networking module.

Contains:
- test_networking_blackbox.py: 246 blackbox tests covering public API behavior
  - Transport: Packet, PacketHeader, Channel types, Connection, UDP, Quality
  - Serialization: BitPacker, Quantizer, DeltaEncoder, NetSerializer
  - Replication: NetGUID, PropertyReplication, Relevancy, Bandwidth, ActorChannel
  - RPC: RPCManager, RPCChannel, RPCValidation
  - Prediction: InputBuffer, Reconciliation, Interpolation, Smoothing
  - Lag Compensation: RewindManager, HitboxHistory, ViewTime
  - Security: Authority, InputValidation, RateLimiter, AnomalyDetector, Response
  - Social: Matchmaking, SkillRating, Lobby, Party, VoiceChat, TextChat

Tests PUBLIC behavior only - no internal state inspection.
"""
