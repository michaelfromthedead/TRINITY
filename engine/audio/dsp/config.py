"""
DSP Configuration Constants

All DSP-related constants for the audio engine's digital signal processing subsystem.
Provides configuration for filters, dynamics, time-based effects, reverb, distortion,
pitch/time manipulation, and special effects.
"""

from typing import Final

# =============================================================================
# General DSP Configuration
# =============================================================================

DEFAULT_SAMPLE_RATE: Final[int] = 48000
"""Default audio sample rate in Hz."""

BLOCK_SIZE: Final[int] = 512
"""Default block size for buffer processing."""

MAX_EFFECT_CHAIN_LENGTH: Final[int] = 16
"""Maximum number of effects in a single chain."""

SIMD_ALIGNMENT: Final[int] = 32
"""Memory alignment for SIMD operations (AVX-256)."""

MAX_CHANNELS: Final[int] = 8
"""Maximum number of audio channels supported."""

# =============================================================================
# Filter Configuration
# =============================================================================

MIN_FREQUENCY: Final[float] = 20.0
"""Minimum filter frequency in Hz (lower bound of human hearing)."""

MAX_FREQUENCY: Final[float] = 20000.0
"""Maximum filter frequency in Hz (upper bound of human hearing)."""

DEFAULT_Q: Final[float] = 0.707
"""Default Q factor (Butterworth response)."""

MIN_Q: Final[float] = 0.1
"""Minimum Q factor."""

MAX_Q: Final[float] = 20.0
"""Maximum Q factor."""

MAX_GAIN_DB: Final[float] = 24.0
"""Maximum gain in dB for shelf and parametric EQ."""

MIN_GAIN_DB: Final[float] = -24.0
"""Minimum gain in dB for shelf and parametric EQ."""

# Biquad coefficient indices
BIQUAD_B0: Final[int] = 0
BIQUAD_B1: Final[int] = 1
BIQUAD_B2: Final[int] = 2
BIQUAD_A1: Final[int] = 3
BIQUAD_A2: Final[int] = 4

# =============================================================================
# Dynamics Processor Configuration
# =============================================================================

# Compressor defaults
COMPRESSOR_DEFAULT_RATIO: Final[float] = 4.0
"""Default compression ratio (4:1)."""

COMPRESSOR_DEFAULT_THRESHOLD_DB: Final[float] = -20.0
"""Default compressor threshold in dB."""

COMPRESSOR_DEFAULT_ATTACK_MS: Final[float] = 0.1
"""Default compressor attack time in milliseconds (very fast for responsive compression)."""

COMPRESSOR_DEFAULT_RELEASE_MS: Final[float] = 100.0
"""Default compressor release time in milliseconds."""

COMPRESSOR_MIN_RATIO: Final[float] = 1.0
"""Minimum compression ratio (no compression)."""

COMPRESSOR_MAX_RATIO: Final[float] = 100.0
"""Maximum compression ratio (near limiting)."""

COMPRESSOR_DEFAULT_KNEE_DB: Final[float] = 6.0
"""Default soft knee width in dB."""

COMPRESSOR_DEFAULT_MAKEUP_DB: Final[float] = 0.0
"""Default makeup gain in dB."""

# Limiter defaults
LIMITER_LOOKAHEAD_MS: Final[float] = 5.0
"""Default limiter lookahead time in milliseconds."""

LIMITER_DEFAULT_CEILING_DB: Final[float] = -0.3
"""Default limiter ceiling in dB."""

LIMITER_DEFAULT_RELEASE_MS: Final[float] = 50.0
"""Default limiter release time in milliseconds."""

# Gate defaults
GATE_DEFAULT_RANGE_DB: Final[float] = -80.0
"""Default gate range in dB (attenuation when closed)."""

GATE_DEFAULT_HOLD_MS: Final[float] = 50.0
"""Default gate hold time in milliseconds."""

GATE_DEFAULT_THRESHOLD_DB: Final[float] = -40.0
"""Default gate threshold in dB."""

GATE_DEFAULT_ATTACK_MS: Final[float] = 0.1
"""Default gate attack time in milliseconds (fast for responsive gating)."""

GATE_DEFAULT_RELEASE_MS: Final[float] = 100.0
"""Default gate release time in milliseconds."""

# Expander defaults
EXPANDER_DEFAULT_RATIO: Final[float] = 2.0
"""Default expansion ratio (1:2)."""

EXPANDER_DEFAULT_THRESHOLD_DB: Final[float] = -30.0
"""Default expander threshold in dB."""

# =============================================================================
# Delay/Time-Based Effects Configuration
# =============================================================================

MAX_DELAY_TIME_MS: Final[float] = 2000.0
"""Maximum delay time in milliseconds."""

MAX_DELAY_FEEDBACK: Final[float] = 0.95
"""Maximum delay feedback to prevent runaway."""

DEFAULT_DELAY_TIME_MS: Final[float] = 250.0
"""Default delay time in milliseconds."""

DEFAULT_DELAY_FEEDBACK: Final[float] = 0.5
"""Default delay feedback."""

DEFAULT_DELAY_WET: Final[float] = 0.5
"""Default delay wet/dry mix."""

# Chorus defaults
CHORUS_DEFAULT_RATE: Final[float] = 1.0
"""Default chorus modulation rate in Hz."""

CHORUS_DEFAULT_DEPTH: Final[float] = 0.5
"""Default chorus modulation depth (0-1)."""

CHORUS_DEFAULT_DELAY_MS: Final[float] = 20.0
"""Default chorus base delay in milliseconds."""

CHORUS_MAX_DELAY_MS: Final[float] = 50.0
"""Maximum chorus delay in milliseconds."""

CHORUS_VOICES: Final[int] = 3
"""Number of chorus voices."""

# Flanger defaults
FLANGER_MAX_DELAY_MS: Final[float] = 20.0
"""Maximum flanger delay in milliseconds."""

FLANGER_DEFAULT_DELAY_MS: Final[float] = 5.0
"""Default flanger base delay in milliseconds."""

FLANGER_DEFAULT_RATE: Final[float] = 0.5
"""Default flanger LFO rate in Hz."""

FLANGER_DEFAULT_DEPTH: Final[float] = 0.7
"""Default flanger modulation depth."""

FLANGER_DEFAULT_FEEDBACK: Final[float] = 0.5
"""Default flanger feedback."""

# Phaser defaults
PHASER_STAGES: Final[int] = 6
"""Number of all-pass stages in phaser."""

PHASER_DEFAULT_RATE: Final[float] = 0.5
"""Default phaser LFO rate in Hz."""

PHASER_DEFAULT_DEPTH: Final[float] = 0.7
"""Default phaser modulation depth."""

PHASER_DEFAULT_FEEDBACK: Final[float] = 0.5
"""Default phaser feedback."""

PHASER_MIN_FREQUENCY: Final[float] = 100.0
"""Minimum phaser all-pass frequency."""

PHASER_MAX_FREQUENCY: Final[float] = 4000.0
"""Maximum phaser all-pass frequency."""

# Vibrato defaults
VIBRATO_DEFAULT_RATE: Final[float] = 5.0
"""Default vibrato rate in Hz."""

VIBRATO_DEFAULT_DEPTH: Final[float] = 0.5
"""Default vibrato depth (0-1)."""

VIBRATO_MIN_RATE: Final[float] = 0.1
"""Minimum vibrato rate in Hz."""

VIBRATO_MAX_RATE: Final[float] = 20.0
"""Maximum vibrato rate in Hz."""

VIBRATO_MAX_DELAY_MS: Final[float] = 20.0
"""Maximum delay time for vibrato effect (ms)."""

# =============================================================================
# Reverb Configuration
# =============================================================================

REVERB_DEFAULT_DECAY_TIME: Final[float] = 2.0
"""Default reverb decay time in seconds."""

REVERB_DEFAULT_ROOM_SIZE: Final[float] = 0.5
"""Default reverb room size (0-1)."""

REVERB_DEFAULT_DAMPING: Final[float] = 0.5
"""Default high-frequency damping (0-1)."""

REVERB_DEFAULT_WET: Final[float] = 0.3
"""Default reverb wet mix (0-1)."""

REVERB_DEFAULT_DRY: Final[float] = 0.7
"""Default reverb dry mix (0-1)."""

REVERB_MAX_PREDELAY_MS: Final[float] = 100.0
"""Maximum reverb pre-delay in milliseconds."""

REVERB_DEFAULT_PREDELAY_MS: Final[float] = 20.0
"""Default reverb pre-delay in milliseconds."""

REVERB_MIN_DECAY_TIME: Final[float] = 0.1
"""Minimum reverb decay time in seconds."""

REVERB_MAX_DECAY_TIME: Final[float] = 30.0
"""Maximum reverb decay time in seconds."""

# Freeverb constants (Schroeder reverb)
REVERB_COMB_DELAYS: Final[tuple] = (1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116)
"""Comb filter delay lengths in samples (at 44.1kHz)."""

REVERB_ALLPASS_DELAYS: Final[tuple] = (225, 556, 441, 341)
"""All-pass filter delay lengths in samples (at 44.1kHz)."""

REVERB_STEREO_SPREAD: Final[int] = 23
"""Stereo spread in samples."""

# =============================================================================
# Distortion Configuration
# =============================================================================

DISTORTION_DEFAULT_DRIVE: Final[float] = 1.0
"""Default distortion drive amount."""

DISTORTION_MIN_DRIVE: Final[float] = 0.0
"""Minimum distortion drive."""

DISTORTION_MAX_DRIVE: Final[float] = 10.0
"""Maximum distortion drive."""

DISTORTION_DEFAULT_OUTPUT_GAIN: Final[float] = 0.7
"""Default distortion output gain."""

# Bitcrusher defaults
BITCRUSH_MIN_BITS: Final[int] = 1
"""Minimum bit depth for bitcrusher."""

BITCRUSH_MAX_BITS: Final[int] = 16
"""Maximum bit depth for bitcrusher."""

BITCRUSH_DEFAULT_BITS: Final[int] = 8
"""Default bit depth for bitcrusher."""

SAMPLE_RATE_REDUCTION_MIN: Final[int] = 1000
"""Minimum sample rate for sample rate reduction."""

SAMPLE_RATE_REDUCTION_DEFAULT: Final[int] = 8000
"""Default reduced sample rate."""

# Waveshaping
WAVESHAPE_TABLE_SIZE: Final[int] = 4096
"""Size of waveshaping lookup table."""

# =============================================================================
# Pitch and Time Manipulation Configuration
# =============================================================================

MAX_PITCH_SHIFT_SEMITONES: Final[float] = 24.0
"""Maximum pitch shift in semitones (+/- 2 octaves)."""

MIN_PITCH_SHIFT_SEMITONES: Final[float] = -24.0
"""Minimum pitch shift in semitones (-2 octaves)."""

MAX_TIME_STRETCH_RATIO: Final[float] = 4.0
"""Maximum time stretch ratio."""

MIN_TIME_STRETCH_RATIO: Final[float] = 0.25
"""Minimum time stretch ratio."""

GRANULAR_GRAIN_SIZE_MS: Final[float] = 50.0
"""Default grain size for granular processing."""

GRANULAR_MIN_GRAIN_SIZE_MS: Final[float] = 10.0
"""Minimum grain size in milliseconds."""

GRANULAR_MAX_GRAIN_SIZE_MS: Final[float] = 200.0
"""Maximum grain size in milliseconds."""

GRANULAR_OVERLAP: Final[float] = 0.5
"""Default grain overlap (0-1)."""

# Phase vocoder
PHASE_VOCODER_FFT_SIZE: Final[int] = 2048
"""FFT size for phase vocoder."""

PHASE_VOCODER_HOP_SIZE: Final[int] = 512
"""Hop size for phase vocoder."""

# =============================================================================
# Special Effects Configuration
# =============================================================================

# Radio effect
RADIO_LOWCUT_FREQ: Final[float] = 300.0
"""Low cut frequency for radio effect."""

RADIO_HIGHCUT_FREQ: Final[float] = 3400.0
"""High cut frequency for radio effect."""

RADIO_DISTORTION_AMOUNT: Final[float] = 0.3
"""Distortion amount for radio effect."""

# Underwater effect
UNDERWATER_CUTOFF_FREQ: Final[float] = 500.0
"""Low-pass cutoff for underwater effect."""

UNDERWATER_RESONANCE: Final[float] = 2.0
"""Resonance Q for underwater effect."""

UNDERWATER_WET_MIX: Final[float] = 0.8
"""Wet mix for underwater effect."""

# Slow motion effect
SLOWMO_PITCH_SEMITONES: Final[float] = -12.0
"""Pitch shift for slow motion effect (1 octave down)."""

SLOWMO_TIME_STRETCH: Final[float] = 2.0
"""Time stretch ratio for slow motion."""

SLOWMO_REVERB_MIX: Final[float] = 0.4
"""Reverb mix for slow motion effect."""

# Explosion effect
EXPLOSION_COMPRESSION_RATIO: Final[float] = 10.0
"""Compression ratio for explosion effect."""

EXPLOSION_DISTORTION_DRIVE: Final[float] = 3.0
"""Distortion drive for explosion effect."""

EXPLOSION_LOWPASS_FREQ: Final[float] = 2000.0
"""Low-pass frequency for explosion rumble."""

EXPLOSION_TINNITUS_FREQ: Final[float] = 4000.0
"""Tinnitus frequency for explosion effect (Hz)."""

EXPLOSION_MUFFLED_FREQ: Final[float] = 500.0
"""Muffled low-pass frequency for explosion effect (Hz)."""

EXPLOSION_RECOVERY_FREQ_MAX: Final[float] = 20000.0
"""Maximum frequency during recovery (Hz)."""

# Muffled effect
MUFFLED_DEFAULT_CUTOFF: Final[float] = 1000.0
"""Default cutoff for muffled effect (Hz)."""

MUFFLED_DEFAULT_REDUCTION_DB: Final[float] = -12.0
"""Default level reduction for muffled effect (dB)."""

# Phone effect
PHONE_LOWCUT_FREQ: Final[float] = 300.0
"""Low cut frequency for phone effect (Hz)."""

PHONE_HIGHCUT_FREQ: Final[float] = 3400.0
"""High cut frequency for phone effect (Hz)."""

# Megaphone effect
MEGAPHONE_CENTER_FREQ: Final[float] = 1000.0
"""Center frequency for megaphone bandpass (Hz)."""

MEGAPHONE_Q: Final[float] = 2.0
"""Q factor for megaphone bandpass."""

MEGAPHONE_DRIVE: Final[float] = 2.0
"""Distortion drive for megaphone effect."""

# Cave effect
CAVE_DEFAULT_DELAY_MS: Final[float] = 150.0
"""Default delay time for cave effect (ms)."""

CAVE_DEFAULT_FEEDBACK: Final[float] = 0.4
"""Default feedback for cave effect."""

CAVE_LOWPASS_FREQ: Final[float] = 8000.0
"""Low-pass frequency for cave effect (Hz)."""

# =============================================================================
# Performance/Quality Settings
# =============================================================================

INTERPOLATION_LINEAR: Final[int] = 0
"""Linear interpolation mode."""

INTERPOLATION_CUBIC: Final[int] = 1
"""Cubic interpolation mode."""

INTERPOLATION_SINC: Final[int] = 2
"""Sinc interpolation mode (highest quality)."""

DEFAULT_INTERPOLATION: Final[int] = INTERPOLATION_CUBIC
"""Default interpolation mode."""

# Envelope follower
ENVELOPE_PEAK: Final[int] = 0
"""Peak detection mode."""

ENVELOPE_RMS: Final[int] = 1
"""RMS detection mode."""

DEFAULT_ENVELOPE_MODE: Final[int] = ENVELOPE_RMS
"""Default envelope detection mode."""

RMS_WINDOW_MS: Final[float] = 10.0
"""RMS window size in milliseconds."""

# =============================================================================
# Sidechain Compression Configuration
# =============================================================================

SIDECHAIN_DEFAULT_RATIO: Final[float] = 4.0
"""Default sidechain compression ratio."""

SIDECHAIN_DEFAULT_THRESHOLD_DB: Final[float] = -20.0
"""Default sidechain compressor threshold in dB."""

SIDECHAIN_DEFAULT_ATTACK_MS: Final[float] = 5.0
"""Default sidechain compressor attack time in milliseconds."""

SIDECHAIN_DEFAULT_RELEASE_MS: Final[float] = 50.0
"""Default sidechain compressor release time in milliseconds."""

SIDECHAIN_DEFAULT_KNEE_DB: Final[float] = 3.0
"""Default sidechain compressor soft knee width in dB."""

SIDECHAIN_DEFAULT_MAKEUP_DB: Final[float] = 0.0
"""Default sidechain compressor makeup gain in dB."""

SIDECHAIN_DEFAULT_MIX: Final[float] = 1.0
"""Default sidechain compressor wet/dry mix (1.0 = fully wet)."""

SIDECHAIN_MIN_RATIO: Final[float] = 1.0
"""Minimum sidechain compression ratio."""

SIDECHAIN_MAX_RATIO: Final[float] = 100.0
"""Maximum sidechain compression ratio."""

# =============================================================================
# Numerical Stability
# =============================================================================

DENORMAL_THRESHOLD: Final[float] = 1e-15
"""Threshold below which values are flushed to zero to prevent denormals."""

DENORMAL_DC_OFFSET: Final[float] = 1e-25
"""Tiny DC offset to prevent denormal numbers in feedback loops."""

DB_FLOOR: Final[float] = -120.0
"""Floor value for dB calculations to prevent -inf."""

EPSILON: Final[float] = 1e-10
"""Small value for preventing division by zero."""

# =============================================================================
# Thread Safety
# =============================================================================

MAX_PARAMETER_SMOOTHING_MS: Final[float] = 20.0
"""Maximum parameter smoothing time in milliseconds."""

PARAMETER_SMOOTHING_DEFAULT_MS: Final[float] = 5.0
"""Default parameter smoothing time in milliseconds."""


def ms_to_samples(ms: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    """Convert milliseconds to samples."""
    return int((ms / 1000.0) * sample_rate)


def samples_to_ms(samples: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Convert samples to milliseconds."""
    return (samples / sample_rate) * 1000.0


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear gain to decibels."""
    import math
    if linear <= 0.0:
        return -120.0  # Effectively -inf dB
    return 20.0 * math.log10(linear)


def frequency_to_normalized(freq: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Convert frequency in Hz to normalized frequency (0-1)."""
    return freq / (sample_rate / 2.0)


def semitones_to_ratio(semitones: float) -> float:
    """Convert semitones to pitch ratio."""
    return 2.0 ** (semitones / 12.0)


def ratio_to_semitones(ratio: float) -> float:
    """Convert pitch ratio to semitones."""
    import math
    if ratio <= 0.0:
        return 0.0
    return 12.0 * math.log2(ratio)
