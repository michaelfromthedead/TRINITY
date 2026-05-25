# PHASE 4 ARCH: Advanced Spatialization

**RDC Phase Architecture**
**Phase**: HRTF, VBAP, Ambisonics, and Surround

---

## Phase Overview

Implement advanced spatialization methods beyond basic stereo: binaural HRTF for headphones, VBAP for arbitrary speaker layouts, Ambisonics for full-sphere encoding, and multichannel surround panning.

---

## Components

### 4.1 HRTFSpatializer

**Purpose**: Binaural audio rendering simulating human head acoustics.

**Key Concepts**:

#### Interaural Time Difference (ITD)
Delay between ears due to head size.

**Woodworth's Formula** (spherical head model):
```
ITD = (r/c) * (theta + sin(theta))
```
Where:
- `r` = head radius (8.75 cm typical)
- `c` = speed of sound (343 m/s)
- `theta` = azimuth in radians

#### Interaural Level Difference (ILD)
Volume difference between ears due to head shadowing.

```
ILD = ILD_MAX_DB * sin(azimuth) * cos(elevation)
```
Higher frequencies shadow more than lower frequencies.

#### HRTF Filters
Synthetic filters simulating pinna (ear) coloration:
- Low-pass on contralateral (far) ear
- Comb filtering from pinna reflections
- Elevation cues from spectral shaping

**HRTFProcessingState**:
- `delay_buffer_left: np.ndarray` - Left ear delay line
- `delay_buffer_right: np.ndarray` - Right ear delay line
- `filter_state_left` - Convolution state
- `filter_state_right` - Convolution state
- `last_azimuth: float` - For interpolation
- `last_elevation: float` - For interpolation

### 4.2 VBAPSpatializer

**Purpose**: Vector Base Amplitude Panning for arbitrary speaker layouts.

**Concept**: Place virtual source between two (2D) or three (3D) speakers by solving for gains that produce the correct direction vector.

**2D VBAP Algorithm**:
1. Convert source direction to Cartesian unit vector
2. Find speaker pair that "brackets" the source direction
3. Solve: `g1 * v1 + g2 * v2 = source_dir`
4. Normalize gains to maintain constant power

**Matrix Solution**:
```
| v1x  v2x | | g1 |   | sx |
| v1y  v2y | | g2 | = | sy |
```
Solve via 2x2 matrix inverse (determinant method).

**Speaker Pair Selection**:
- Pre-compute speaker pairs for each angular sector
- At runtime, find sector containing source direction
- Fall back to nearest speaker if out of range

### 4.3 AmbisonicsSpatializer

**Purpose**: Spherical harmonic encoding for renderer-agnostic spatial audio.

**First-Order B-Format** (ACN ordering):
- `W` = omnidirectional (0th order)
- `Y` = front-back gradient
- `Z` = up-down gradient  
- `X` = left-right gradient

**Encoding**:
```python
W = gain / sqrt(2)  # Omni (scaled for energy)
Y = gain * sin(azimuth) * cos(elevation)
Z = gain * sin(elevation)
X = gain * cos(azimuth) * cos(elevation)
```

**Decoding**:
Generate decoder matrix from speaker positions:
```python
for each speaker at (az, el):
    D[speaker][W] = 1 / sqrt(2)
    D[speaker][Y] = sin(az) * cos(el)
    D[speaker][Z] = sin(el)
    D[speaker][X] = cos(az) * cos(el)
```
Output = D * B-format input

**Spread Parameter**:
Reduce directional components (Y, Z, X) to spread source:
```python
directional_scale = 1.0 - spread
Y *= directional_scale
Z *= directional_scale
X *= directional_scale
```

### 4.4 SurroundPanner

**Purpose**: Channel routing for standard surround layouts (5.1, 7.1, Atmos).

**5.1 Layout**:
- L (Left), R (Right), C (Center)
- LFE (Subwoofer)
- Ls (Left Surround), Rs (Right Surround)

**7.1 Layout**:
- Add: Lrs (Left Rear), Rrs (Right Rear)

**Atmos Layout**:
- Add height channels: Ltf, Rtf, Ltr, Rtr

**Routing Logic**:
1. Calculate azimuth and elevation to source
2. Map to nearest speaker pair/triplet
3. Apply VBAP-like gain distribution
4. Route low frequencies to LFE with crossover

---

## Speaker Configuration

**File**: `engine/audio/spatial/speaker_config.py`

**SpeakerLayout**:
- `name: str` - Layout identifier
- `speakers: List[Speaker]` - Speaker definitions
- `pairs: List[Tuple[int, int]]` - Pre-computed VBAP pairs
- `channel_map: Dict[str, int]` - Name to index

**Speaker**:
- `name: str` - L, R, C, etc.
- `azimuth: float` - Horizontal angle
- `elevation: float` - Vertical angle
- `distance: float` - From listener

---

## Data Flow

```
Source Direction (azimuth, elevation)
              |
              v
   +--------------------+
   | Select Spatializer |
   +--------------------+
              |
    +---------+---------+---------+
    |         |         |         |
    v         v         v         v
 Stereo     VBAP    Ambisonics  HRTF
    |         |         |         |
    v         v         v         v
 2 gains   N gains   4 B-fmt   2 filtered
    |         |         |         |
    +----+----+----+----+         |
         |                        |
         v                        v
   Speaker Feeds            Headphone Out
```

---

## Configuration

- `HRTF_SAMPLE_RATE = 48000`
- `HRTF_FILTER_LENGTH = 128`
- `VBAP_MAX_SPEAKERS = 24`
- `AMBISONICS_ORDER = 1` (first-order)

---

## Success Criteria

1. HRTF produces convincing externalization in headphones
2. ITD calculation matches Woodworth formula
3. VBAP places sounds accurately between speakers
4. Ambisonics encodes/decodes without gain changes
5. Surround panning routes correctly to all channels
