"""RPC (Remote Procedure Call) system for networked game actions.

This module provides remote procedure call functionality for communication
between server and clients, including authority validation and rate limiting.

Components:
    - RPCManager: Registration and dispatch of remote procedure calls
    - RPCChannel: Network transmission of RPC messages
    - RPCValidation: Authority checking and rate limiting
"""

from .rpc_manager import (
    RPCManager,
    RPCInfo,
    RPCAuthority,
    RPCReliability,
    RPCCallResult,
    PendingRPC,
    rpc,
)

from .rpc_channel import (
    RPCChannel,
    RPCChannelManager,
    RPCChannelState,
    RPCMessage,
)

from .rpc_validation import (
    RPCValidator,
    RateLimiter,
    RateLimitConfig,
    ValidationError,
    validate_authority,
    validate_rate_limit,
    validate_param_range,
    validate_param_type,
    validate_param_length,
)

__all__ = [
    # RPC Manager
    'RPCManager',
    'RPCInfo',
    'RPCAuthority',
    'RPCReliability',
    'RPCCallResult',
    'PendingRPC',
    'rpc',

    # RPC Channel
    'RPCChannel',
    'RPCChannelManager',
    'RPCChannelState',
    'RPCMessage',

    # Validation
    'RPCValidator',
    'RateLimiter',
    'RateLimitConfig',
    'ValidationError',
    'validate_authority',
    'validate_rate_limit',
    'validate_param_range',
    'validate_param_type',
    'validate_param_length',
]
