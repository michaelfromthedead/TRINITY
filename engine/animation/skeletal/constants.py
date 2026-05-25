"""Animation skeletal subsystem constants.

All magic numbers for skinning, root motion, retargeting, and compression.
"""

# === Weight Normalization ===
WEIGHT_NORMALIZATION_EPSILON: float = 1e-9
DEFAULT_BONE_WEIGHT: float = 1.0
MAX_BONE_INFLUENCES: int = 4

# === Compression Bit Depths ===
COMPRESSION_BITS_LOW: int = 8
COMPRESSION_BITS_MEDIUM: int = 16
COMPRESSION_BITS_HIGH: int = 32
DEFAULT_TRANSLATION_BITS: int = 16
DEFAULT_ROTATION_BITS: int = 16
DEFAULT_SCALE_BITS: int = 16
MIN_COMPRESSION_BITS: int = 8

# === Compression Error Thresholds ===
DEFAULT_TRANSLATION_ERROR_THRESHOLD: float = 0.001  # 1mm precision
DEFAULT_ROTATION_ERROR_THRESHOLD: float = 0.0001  # ~0.006 degrees
DEFAULT_SCALE_ERROR_THRESHOLD: float = 0.0001
DEFAULT_CURVE_FITTING_TOLERANCE: float = 0.001

# === Quantization Range Padding ===
QUANTIZATION_RANGE_PADDING_FACTOR: float = 0.001
QUANTIZATION_MIN_PADDING: float = 0.001

# === Retargeting ===
RETARGET_POSITION_MATCH_THRESHOLD: float = 0.5
DEFAULT_UNMAPPED_BLEND_FACTOR: float = 0.5
DEFAULT_SCALE_FACTOR: float = 1.0

# === Root Motion ===
DEFAULT_ROOT_MOTION_SCALE: float = 1.0
DEFAULT_ROTATION_SCALE: float = 1.0
DEFAULT_GROUND_HEIGHT: float = 0.0

# === Animation Playback ===
DEFAULT_FRAME_RATE: float = 30.0
DEFAULT_SAMPLE_RATE: float = 0.0  # 0 = keep original

# === Buffer Sizes ===
COMPRESSION_HEADER_SIZE_ESTIMATE: int = 64
CONSTANT_VALUE_STORAGE_SIZE: int = 16

# === Time Constants ===
TIME_EPSILON: float = 1e-9
