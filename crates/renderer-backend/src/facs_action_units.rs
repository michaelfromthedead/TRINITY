//! FACS Action Units System for TRINITY Engine (T-AN-7.2).
//!
//! This module implements the Facial Action Coding System (FACS) for
//! scientific and artistic facial animation control:
//!
//! - 18+ core action units with left/right variants
//! - FACS A-E intensity scale (0.0 - 5.0)
//! - Asymmetry control for realistic expressions
//! - Expression presets (6 basic emotions + more)
//! - AU combination rules with conflict detection
//! - Smooth transitions with decay curves
//! - Integration with blend_shapes.rs
//!
//! # FACS Background
//!
//! FACS (Facial Action Coding System) was developed by Paul Ekman and
//! Wallace V. Friesen to systematically describe facial movements.
//! Each Action Unit (AU) represents a specific muscular action.
//!
//! # Intensity Scale (A-E)
//!
//! | Level | Value | Description          |
//! |-------|-------|----------------------|
//! | A     | 1.0   | Trace                |
//! | B     | 2.0   | Slight               |
//! | C     | 3.0   | Marked/Pronounced    |
//! | D     | 4.0   | Severe/Extreme       |
//! | E     | 5.0   | Maximum              |
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::facs_action_units::{
//!     FACSController, ActionUnit, ActionUnitWeight, ExpressionPreset,
//! };
//! use renderer_backend::blend_shapes::BlendShapeSet;
//!
//! // Create a FACS controller
//! let mut controller = FACSController::new();
//!
//! // Set individual action units
//! controller.set_au(ActionUnit::AU12, 3.0); // Smile at intensity C
//! controller.set_au(ActionUnit::AU6, 2.5);  // Cheek raiser
//!
//! // Or use expression presets
//! controller.apply_preset(ExpressionPreset::Joy);
//!
//! // Apply to blend shapes
//! let blend_weights = controller.to_blend_weights();
//! ```

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::blend_shapes::{BlendShapeSet, FaceRegion};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum intensity value (trace).
pub const INTENSITY_MIN: f32 = 0.0;

/// Maximum intensity value (maximum/extreme).
pub const INTENSITY_MAX: f32 = 5.0;

/// FACS intensity level A (trace).
pub const INTENSITY_A: f32 = 1.0;

/// FACS intensity level B (slight).
pub const INTENSITY_B: f32 = 2.0;

/// FACS intensity level C (marked/pronounced).
pub const INTENSITY_C: f32 = 3.0;

/// FACS intensity level D (severe/extreme).
pub const INTENSITY_D: f32 = 4.0;

/// FACS intensity level E (maximum).
pub const INTENSITY_E: f32 = 5.0;

/// Weight threshold for considering an AU "active".
pub const AU_WEIGHT_THRESHOLD: f32 = 0.001;

/// Maximum combined intensity when stacking multiple AUs.
pub const MAX_STACKED_INTENSITY: f32 = 6.0;

/// Default decay rate per second for natural falloff.
pub const DEFAULT_DECAY_RATE: f32 = 2.0;

/// Default transition duration in seconds.
pub const DEFAULT_TRANSITION_DURATION: f32 = 0.25;

// ---------------------------------------------------------------------------
// ActionUnit
// ---------------------------------------------------------------------------

/// FACS Action Unit identifiers.
///
/// Each AU represents a specific facial muscle movement.
/// Bilateral AUs (left/right) are provided for asymmetric expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionUnit {
    // Upper Face - Brow Region
    /// AU1: Inner Brow Raiser (frontalis, pars medialis).
    AU1,
    /// AU1L: Inner Brow Raiser (left side).
    AU1L,
    /// AU1R: Inner Brow Raiser (right side).
    AU1R,

    /// AU2: Outer Brow Raiser (frontalis, pars lateralis).
    AU2,
    /// AU2L: Outer Brow Raiser (left side).
    AU2L,
    /// AU2R: Outer Brow Raiser (right side).
    AU2R,

    /// AU4: Brow Lowerer (corrugator supercilii, depressor supercilii).
    AU4,
    /// AU4L: Brow Lowerer (left side).
    AU4L,
    /// AU4R: Brow Lowerer (right side).
    AU4R,

    // Upper Face - Eye Region
    /// AU5: Upper Lid Raiser (levator palpebrae superioris).
    AU5,
    /// AU5L: Upper Lid Raiser (left side).
    AU5L,
    /// AU5R: Upper Lid Raiser (right side).
    AU5R,

    /// AU6: Cheek Raiser (orbicularis oculi, pars orbitalis).
    AU6,
    /// AU6L: Cheek Raiser (left side).
    AU6L,
    /// AU6R: Cheek Raiser (right side).
    AU6R,

    /// AU7: Lid Tightener (orbicularis oculi, pars palpebralis).
    AU7,
    /// AU7L: Lid Tightener (left side).
    AU7L,
    /// AU7R: Lid Tightener (right side).
    AU7R,

    /// AU43: Eyes Closed (relaxation of levator palpebrae superioris).
    AU43,
    /// AU43L: Eyes Closed (left side).
    AU43L,
    /// AU43R: Eyes Closed (right side).
    AU43R,

    /// AU45: Blink (relaxation of levator palpebrae).
    AU45,
    /// AU45L: Blink (left side).
    AU45L,
    /// AU45R: Blink (right side).
    AU45R,

    // Lower Face - Nose Region
    /// AU9: Nose Wrinkler (levator labii superioris alaeque nasi).
    AU9,
    /// AU9L: Nose Wrinkler (left side).
    AU9L,
    /// AU9R: Nose Wrinkler (right side).
    AU9R,

    // Lower Face - Mouth Region
    /// AU10: Upper Lip Raiser (levator labii superioris).
    AU10,
    /// AU10L: Upper Lip Raiser (left side).
    AU10L,
    /// AU10R: Upper Lip Raiser (right side).
    AU10R,

    /// AU12: Lip Corner Puller - Smile (zygomaticus major).
    AU12,
    /// AU12L: Lip Corner Puller (left side).
    AU12L,
    /// AU12R: Lip Corner Puller (right side).
    AU12R,

    /// AU15: Lip Corner Depressor (depressor anguli oris).
    AU15,
    /// AU15L: Lip Corner Depressor (left side).
    AU15L,
    /// AU15R: Lip Corner Depressor (right side).
    AU15R,

    /// AU17: Chin Raiser (mentalis).
    AU17,

    /// AU20: Lip Stretcher (risorius).
    AU20,
    /// AU20L: Lip Stretcher (left side).
    AU20L,
    /// AU20R: Lip Stretcher (right side).
    AU20R,

    /// AU23: Lip Tightener (orbicularis oris).
    AU23,

    /// AU25: Lips Part (depressor labii inferioris, or relaxation).
    AU25,

    /// AU26: Jaw Drop (masseter relaxation; depressor muscles).
    AU26,

    /// AU27: Mouth Stretch (pterygoids, digastric).
    AU27,
}

impl ActionUnit {
    /// Get all base action units (without left/right variants).
    pub fn all_base() -> &'static [ActionUnit] {
        &[
            ActionUnit::AU1,
            ActionUnit::AU2,
            ActionUnit::AU4,
            ActionUnit::AU5,
            ActionUnit::AU6,
            ActionUnit::AU7,
            ActionUnit::AU9,
            ActionUnit::AU10,
            ActionUnit::AU12,
            ActionUnit::AU15,
            ActionUnit::AU17,
            ActionUnit::AU20,
            ActionUnit::AU23,
            ActionUnit::AU25,
            ActionUnit::AU26,
            ActionUnit::AU27,
            ActionUnit::AU43,
            ActionUnit::AU45,
        ]
    }

    /// Get all action units including left/right variants.
    pub fn all() -> &'static [ActionUnit] {
        &[
            ActionUnit::AU1,
            ActionUnit::AU1L,
            ActionUnit::AU1R,
            ActionUnit::AU2,
            ActionUnit::AU2L,
            ActionUnit::AU2R,
            ActionUnit::AU4,
            ActionUnit::AU4L,
            ActionUnit::AU4R,
            ActionUnit::AU5,
            ActionUnit::AU5L,
            ActionUnit::AU5R,
            ActionUnit::AU6,
            ActionUnit::AU6L,
            ActionUnit::AU6R,
            ActionUnit::AU7,
            ActionUnit::AU7L,
            ActionUnit::AU7R,
            ActionUnit::AU9,
            ActionUnit::AU9L,
            ActionUnit::AU9R,
            ActionUnit::AU10,
            ActionUnit::AU10L,
            ActionUnit::AU10R,
            ActionUnit::AU12,
            ActionUnit::AU12L,
            ActionUnit::AU12R,
            ActionUnit::AU15,
            ActionUnit::AU15L,
            ActionUnit::AU15R,
            ActionUnit::AU17,
            ActionUnit::AU20,
            ActionUnit::AU20L,
            ActionUnit::AU20R,
            ActionUnit::AU23,
            ActionUnit::AU25,
            ActionUnit::AU26,
            ActionUnit::AU27,
            ActionUnit::AU43,
            ActionUnit::AU43L,
            ActionUnit::AU43R,
            ActionUnit::AU45,
            ActionUnit::AU45L,
            ActionUnit::AU45R,
        ]
    }

    /// Get the FACS code string (e.g., "AU1", "AU12L").
    pub fn code(&self) -> &'static str {
        match self {
            ActionUnit::AU1 => "AU1",
            ActionUnit::AU1L => "AU1L",
            ActionUnit::AU1R => "AU1R",
            ActionUnit::AU2 => "AU2",
            ActionUnit::AU2L => "AU2L",
            ActionUnit::AU2R => "AU2R",
            ActionUnit::AU4 => "AU4",
            ActionUnit::AU4L => "AU4L",
            ActionUnit::AU4R => "AU4R",
            ActionUnit::AU5 => "AU5",
            ActionUnit::AU5L => "AU5L",
            ActionUnit::AU5R => "AU5R",
            ActionUnit::AU6 => "AU6",
            ActionUnit::AU6L => "AU6L",
            ActionUnit::AU6R => "AU6R",
            ActionUnit::AU7 => "AU7",
            ActionUnit::AU7L => "AU7L",
            ActionUnit::AU7R => "AU7R",
            ActionUnit::AU9 => "AU9",
            ActionUnit::AU9L => "AU9L",
            ActionUnit::AU9R => "AU9R",
            ActionUnit::AU10 => "AU10",
            ActionUnit::AU10L => "AU10L",
            ActionUnit::AU10R => "AU10R",
            ActionUnit::AU12 => "AU12",
            ActionUnit::AU12L => "AU12L",
            ActionUnit::AU12R => "AU12R",
            ActionUnit::AU15 => "AU15",
            ActionUnit::AU15L => "AU15L",
            ActionUnit::AU15R => "AU15R",
            ActionUnit::AU17 => "AU17",
            ActionUnit::AU20 => "AU20",
            ActionUnit::AU20L => "AU20L",
            ActionUnit::AU20R => "AU20R",
            ActionUnit::AU23 => "AU23",
            ActionUnit::AU25 => "AU25",
            ActionUnit::AU26 => "AU26",
            ActionUnit::AU27 => "AU27",
            ActionUnit::AU43 => "AU43",
            ActionUnit::AU43L => "AU43L",
            ActionUnit::AU43R => "AU43R",
            ActionUnit::AU45 => "AU45",
            ActionUnit::AU45L => "AU45L",
            ActionUnit::AU45R => "AU45R",
        }
    }

    /// Get a human-readable name for this action unit.
    pub fn name(&self) -> &'static str {
        match self {
            ActionUnit::AU1 | ActionUnit::AU1L | ActionUnit::AU1R => "Inner Brow Raiser",
            ActionUnit::AU2 | ActionUnit::AU2L | ActionUnit::AU2R => "Outer Brow Raiser",
            ActionUnit::AU4 | ActionUnit::AU4L | ActionUnit::AU4R => "Brow Lowerer",
            ActionUnit::AU5 | ActionUnit::AU5L | ActionUnit::AU5R => "Upper Lid Raiser",
            ActionUnit::AU6 | ActionUnit::AU6L | ActionUnit::AU6R => "Cheek Raiser",
            ActionUnit::AU7 | ActionUnit::AU7L | ActionUnit::AU7R => "Lid Tightener",
            ActionUnit::AU9 | ActionUnit::AU9L | ActionUnit::AU9R => "Nose Wrinkler",
            ActionUnit::AU10 | ActionUnit::AU10L | ActionUnit::AU10R => "Upper Lip Raiser",
            ActionUnit::AU12 | ActionUnit::AU12L | ActionUnit::AU12R => "Lip Corner Puller",
            ActionUnit::AU15 | ActionUnit::AU15L | ActionUnit::AU15R => "Lip Corner Depressor",
            ActionUnit::AU17 => "Chin Raiser",
            ActionUnit::AU20 | ActionUnit::AU20L | ActionUnit::AU20R => "Lip Stretcher",
            ActionUnit::AU23 => "Lip Tightener",
            ActionUnit::AU25 => "Lips Part",
            ActionUnit::AU26 => "Jaw Drop",
            ActionUnit::AU27 => "Mouth Stretch",
            ActionUnit::AU43 | ActionUnit::AU43L | ActionUnit::AU43R => "Eyes Closed",
            ActionUnit::AU45 | ActionUnit::AU45L | ActionUnit::AU45R => "Blink",
        }
    }

    /// Get the affected face region for this action unit.
    pub fn face_region(&self) -> FaceRegion {
        match self {
            ActionUnit::AU1 | ActionUnit::AU2 | ActionUnit::AU4 => FaceRegion::LeftBrow,
            ActionUnit::AU1L | ActionUnit::AU2L | ActionUnit::AU4L => FaceRegion::LeftBrow,
            ActionUnit::AU1R | ActionUnit::AU2R | ActionUnit::AU4R => FaceRegion::RightBrow,
            ActionUnit::AU5 | ActionUnit::AU6 | ActionUnit::AU7 | ActionUnit::AU43 | ActionUnit::AU45 => {
                FaceRegion::LeftEye
            }
            ActionUnit::AU5L | ActionUnit::AU6L | ActionUnit::AU7L | ActionUnit::AU43L | ActionUnit::AU45L => {
                FaceRegion::LeftEye
            }
            ActionUnit::AU5R | ActionUnit::AU6R | ActionUnit::AU7R | ActionUnit::AU43R | ActionUnit::AU45R => {
                FaceRegion::RightEye
            }
            ActionUnit::AU9 | ActionUnit::AU9L | ActionUnit::AU9R => FaceRegion::Nose,
            _ => FaceRegion::Mouth,
        }
    }

    /// Check if this is a bilateral (symmetric) action unit.
    pub fn is_bilateral(&self) -> bool {
        matches!(
            self,
            ActionUnit::AU1
                | ActionUnit::AU2
                | ActionUnit::AU4
                | ActionUnit::AU5
                | ActionUnit::AU6
                | ActionUnit::AU7
                | ActionUnit::AU9
                | ActionUnit::AU10
                | ActionUnit::AU12
                | ActionUnit::AU15
                | ActionUnit::AU20
                | ActionUnit::AU43
                | ActionUnit::AU45
        )
    }

    /// Check if this is a left-side variant.
    pub fn is_left(&self) -> bool {
        matches!(
            self,
            ActionUnit::AU1L
                | ActionUnit::AU2L
                | ActionUnit::AU4L
                | ActionUnit::AU5L
                | ActionUnit::AU6L
                | ActionUnit::AU7L
                | ActionUnit::AU9L
                | ActionUnit::AU10L
                | ActionUnit::AU12L
                | ActionUnit::AU15L
                | ActionUnit::AU20L
                | ActionUnit::AU43L
                | ActionUnit::AU45L
        )
    }

    /// Check if this is a right-side variant.
    pub fn is_right(&self) -> bool {
        matches!(
            self,
            ActionUnit::AU1R
                | ActionUnit::AU2R
                | ActionUnit::AU4R
                | ActionUnit::AU5R
                | ActionUnit::AU6R
                | ActionUnit::AU7R
                | ActionUnit::AU9R
                | ActionUnit::AU10R
                | ActionUnit::AU12R
                | ActionUnit::AU15R
                | ActionUnit::AU20R
                | ActionUnit::AU43R
                | ActionUnit::AU45R
        )
    }

    /// Get the left variant of a bilateral AU.
    pub fn left_variant(&self) -> Option<ActionUnit> {
        match self {
            ActionUnit::AU1 => Some(ActionUnit::AU1L),
            ActionUnit::AU2 => Some(ActionUnit::AU2L),
            ActionUnit::AU4 => Some(ActionUnit::AU4L),
            ActionUnit::AU5 => Some(ActionUnit::AU5L),
            ActionUnit::AU6 => Some(ActionUnit::AU6L),
            ActionUnit::AU7 => Some(ActionUnit::AU7L),
            ActionUnit::AU9 => Some(ActionUnit::AU9L),
            ActionUnit::AU10 => Some(ActionUnit::AU10L),
            ActionUnit::AU12 => Some(ActionUnit::AU12L),
            ActionUnit::AU15 => Some(ActionUnit::AU15L),
            ActionUnit::AU20 => Some(ActionUnit::AU20L),
            ActionUnit::AU43 => Some(ActionUnit::AU43L),
            ActionUnit::AU45 => Some(ActionUnit::AU45L),
            _ => None,
        }
    }

    /// Get the right variant of a bilateral AU.
    pub fn right_variant(&self) -> Option<ActionUnit> {
        match self {
            ActionUnit::AU1 => Some(ActionUnit::AU1R),
            ActionUnit::AU2 => Some(ActionUnit::AU2R),
            ActionUnit::AU4 => Some(ActionUnit::AU4R),
            ActionUnit::AU5 => Some(ActionUnit::AU5R),
            ActionUnit::AU6 => Some(ActionUnit::AU6R),
            ActionUnit::AU7 => Some(ActionUnit::AU7R),
            ActionUnit::AU9 => Some(ActionUnit::AU9R),
            ActionUnit::AU10 => Some(ActionUnit::AU10R),
            ActionUnit::AU12 => Some(ActionUnit::AU12R),
            ActionUnit::AU15 => Some(ActionUnit::AU15R),
            ActionUnit::AU20 => Some(ActionUnit::AU20R),
            ActionUnit::AU43 => Some(ActionUnit::AU43R),
            ActionUnit::AU45 => Some(ActionUnit::AU45R),
            _ => None,
        }
    }

    /// Get the base (non-lateralized) version of this AU.
    pub fn base(&self) -> ActionUnit {
        match self {
            ActionUnit::AU1L | ActionUnit::AU1R => ActionUnit::AU1,
            ActionUnit::AU2L | ActionUnit::AU2R => ActionUnit::AU2,
            ActionUnit::AU4L | ActionUnit::AU4R => ActionUnit::AU4,
            ActionUnit::AU5L | ActionUnit::AU5R => ActionUnit::AU5,
            ActionUnit::AU6L | ActionUnit::AU6R => ActionUnit::AU6,
            ActionUnit::AU7L | ActionUnit::AU7R => ActionUnit::AU7,
            ActionUnit::AU9L | ActionUnit::AU9R => ActionUnit::AU9,
            ActionUnit::AU10L | ActionUnit::AU10R => ActionUnit::AU10,
            ActionUnit::AU12L | ActionUnit::AU12R => ActionUnit::AU12,
            ActionUnit::AU15L | ActionUnit::AU15R => ActionUnit::AU15,
            ActionUnit::AU20L | ActionUnit::AU20R => ActionUnit::AU20,
            ActionUnit::AU43L | ActionUnit::AU43R => ActionUnit::AU43,
            ActionUnit::AU45L | ActionUnit::AU45R => ActionUnit::AU45,
            _ => *self,
        }
    }

    /// Get the typical blend shape name for this action unit.
    pub fn blend_shape_name(&self) -> &'static str {
        match self {
            ActionUnit::AU1 => "browInnerUp",
            ActionUnit::AU1L => "browInnerUp_L",
            ActionUnit::AU1R => "browInnerUp_R",
            ActionUnit::AU2 => "browOuterUp",
            ActionUnit::AU2L => "browOuterUp_L",
            ActionUnit::AU2R => "browOuterUp_R",
            ActionUnit::AU4 => "browDown",
            ActionUnit::AU4L => "browDown_L",
            ActionUnit::AU4R => "browDown_R",
            ActionUnit::AU5 => "eyeWideOpen",
            ActionUnit::AU5L => "eyeWideOpen_L",
            ActionUnit::AU5R => "eyeWideOpen_R",
            ActionUnit::AU6 => "cheekRaise",
            ActionUnit::AU6L => "cheekRaise_L",
            ActionUnit::AU6R => "cheekRaise_R",
            ActionUnit::AU7 => "lidTightener",
            ActionUnit::AU7L => "lidTightener_L",
            ActionUnit::AU7R => "lidTightener_R",
            ActionUnit::AU9 => "noseWrinkle",
            ActionUnit::AU9L => "noseWrinkle_L",
            ActionUnit::AU9R => "noseWrinkle_R",
            ActionUnit::AU10 => "upperLipRaise",
            ActionUnit::AU10L => "upperLipRaise_L",
            ActionUnit::AU10R => "upperLipRaise_R",
            ActionUnit::AU12 => "mouthSmile",
            ActionUnit::AU12L => "mouthSmile_L",
            ActionUnit::AU12R => "mouthSmile_R",
            ActionUnit::AU15 => "mouthFrown",
            ActionUnit::AU15L => "mouthFrown_L",
            ActionUnit::AU15R => "mouthFrown_R",
            ActionUnit::AU17 => "chinRaise",
            ActionUnit::AU20 => "lipStretcher",
            ActionUnit::AU20L => "lipStretcher_L",
            ActionUnit::AU20R => "lipStretcher_R",
            ActionUnit::AU23 => "lipTightener",
            ActionUnit::AU25 => "lipsPart",
            ActionUnit::AU26 => "jawDrop",
            ActionUnit::AU27 => "mouthStretch",
            ActionUnit::AU43 => "eyesClosed",
            ActionUnit::AU43L => "eyeClosed_L",
            ActionUnit::AU43R => "eyeClosed_R",
            ActionUnit::AU45 => "blink",
            ActionUnit::AU45L => "blink_L",
            ActionUnit::AU45R => "blink_R",
        }
    }
}

impl fmt::Display for ActionUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.code(), self.name())
    }
}

impl Default for ActionUnit {
    fn default() -> Self {
        ActionUnit::AU12
    }
}

// ---------------------------------------------------------------------------
// ActionUnitWeight
// ---------------------------------------------------------------------------

/// Weight/intensity configuration for a single action unit.
///
/// Includes intensity (FACS A-E scale), asymmetry control, and timing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionUnitWeight {
    /// The action unit being controlled.
    pub action_unit: ActionUnit,
    /// Intensity level (0.0 - 5.0, using FACS A-E scale).
    pub intensity: f32,
    /// Asymmetry factor (-1.0 to 1.0).
    /// - 0.0: symmetric (equal on both sides)
    /// - -1.0: fully left-biased
    /// - 1.0: fully right-biased
    pub asymmetry: f32,
    /// Duration this AU should be held (seconds), 0 = indefinite.
    pub duration: f32,
    /// Time elapsed since activation (seconds).
    pub elapsed: f32,
    /// Decay rate per second for natural falloff.
    pub decay_rate: f32,
}

impl ActionUnitWeight {
    /// Create a new AU weight with the given intensity.
    pub fn new(action_unit: ActionUnit, intensity: f32) -> Self {
        Self {
            action_unit,
            intensity: intensity.clamp(INTENSITY_MIN, INTENSITY_MAX),
            asymmetry: 0.0,
            duration: 0.0,
            elapsed: 0.0,
            decay_rate: 0.0,
        }
    }

    /// Builder: set asymmetry.
    pub fn with_asymmetry(mut self, asymmetry: f32) -> Self {
        self.asymmetry = asymmetry.clamp(-1.0, 1.0);
        self
    }

    /// Builder: set duration.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration.max(0.0);
        self
    }

    /// Builder: set decay rate.
    pub fn with_decay(mut self, decay_rate: f32) -> Self {
        self.decay_rate = decay_rate.max(0.0);
        self
    }

    /// Check if this weight is active (above threshold).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.intensity > AU_WEIGHT_THRESHOLD
    }

    /// Check if this weight has expired based on duration.
    #[inline]
    pub fn is_expired(&self) -> bool {
        self.duration > 0.0 && self.elapsed >= self.duration
    }

    /// Update the weight with delta time, applying decay.
    pub fn update(&mut self, dt: f32) {
        self.elapsed += dt;

        // Apply decay
        if self.decay_rate > 0.0 {
            self.intensity -= self.decay_rate * dt;
            self.intensity = self.intensity.max(0.0);
        }
    }

    /// Get the effective left-side intensity.
    pub fn left_intensity(&self) -> f32 {
        let base = self.intensity / INTENSITY_MAX;
        if self.asymmetry <= 0.0 {
            // Left-biased or symmetric
            base * (1.0 + self.asymmetry.abs())
        } else {
            // Right-biased
            base * (1.0 - self.asymmetry)
        }
        .clamp(0.0, 1.0)
    }

    /// Get the effective right-side intensity.
    pub fn right_intensity(&self) -> f32 {
        let base = self.intensity / INTENSITY_MAX;
        if self.asymmetry >= 0.0 {
            // Right-biased or symmetric
            base * (1.0 + self.asymmetry.abs())
        } else {
            // Left-biased
            base * (1.0 + self.asymmetry)
        }
        .clamp(0.0, 1.0)
    }

    /// Get the normalized intensity (0.0 - 1.0).
    #[inline]
    pub fn normalized_intensity(&self) -> f32 {
        self.intensity / INTENSITY_MAX
    }

    /// Get the FACS intensity level (A-E).
    pub fn facs_level(&self) -> FACSIntensityLevel {
        FACSIntensityLevel::from_value(self.intensity)
    }
}

impl Default for ActionUnitWeight {
    fn default() -> Self {
        Self {
            action_unit: ActionUnit::default(),
            intensity: 0.0,
            asymmetry: 0.0,
            duration: 0.0,
            elapsed: 0.0,
            decay_rate: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// FACSIntensityLevel
// ---------------------------------------------------------------------------

/// FACS intensity level classification (A-E scale).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FACSIntensityLevel {
    /// No visible action.
    None,
    /// A: Trace (just barely visible).
    A,
    /// B: Slight (visible but not pronounced).
    B,
    /// C: Marked/Pronounced (clearly visible).
    C,
    /// D: Severe/Extreme (very strong).
    D,
    /// E: Maximum (extreme, highest intensity).
    E,
}

impl FACSIntensityLevel {
    /// Convert from a numeric intensity value.
    pub fn from_value(value: f32) -> Self {
        if value < 0.5 {
            FACSIntensityLevel::None
        } else if value < 1.5 {
            FACSIntensityLevel::A
        } else if value < 2.5 {
            FACSIntensityLevel::B
        } else if value < 3.5 {
            FACSIntensityLevel::C
        } else if value < 4.5 {
            FACSIntensityLevel::D
        } else {
            FACSIntensityLevel::E
        }
    }

    /// Convert to a representative numeric value.
    pub fn to_value(&self) -> f32 {
        match self {
            FACSIntensityLevel::None => 0.0,
            FACSIntensityLevel::A => INTENSITY_A,
            FACSIntensityLevel::B => INTENSITY_B,
            FACSIntensityLevel::C => INTENSITY_C,
            FACSIntensityLevel::D => INTENSITY_D,
            FACSIntensityLevel::E => INTENSITY_E,
        }
    }

    /// Get the character code (A-E or '-' for none).
    pub fn code(&self) -> char {
        match self {
            FACSIntensityLevel::None => '-',
            FACSIntensityLevel::A => 'A',
            FACSIntensityLevel::B => 'B',
            FACSIntensityLevel::C => 'C',
            FACSIntensityLevel::D => 'D',
            FACSIntensityLevel::E => 'E',
        }
    }
}

impl Default for FACSIntensityLevel {
    fn default() -> Self {
        FACSIntensityLevel::None
    }
}

impl fmt::Display for FACSIntensityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

// ---------------------------------------------------------------------------
// ExpressionPreset
// ---------------------------------------------------------------------------

/// Pre-defined expression presets based on FACS research.
///
/// These combine multiple action units to create recognizable expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpressionPreset {
    /// Neutral/relaxed expression.
    Neutral,

    // Basic Emotions (Ekman's 6)
    /// Joy/Happiness: AU6 + AU12 (Duchenne smile).
    Joy,
    /// Sadness: AU1 + AU4 + AU15.
    Sadness,
    /// Anger: AU4 + AU5 + AU7 + AU23.
    Anger,
    /// Fear: AU1 + AU2 + AU4 + AU5 + AU20 + AU26.
    Fear,
    /// Disgust: AU9 + AU15 + AU17.
    Disgust,
    /// Surprise: AU1 + AU2 + AU5 + AU26.
    Surprise,

    // Additional expressions
    /// Contempt: AU12 + AU14 (unilateral).
    Contempt,
    /// Confusion: AU4 + AU1 + AU7.
    Confusion,
    /// Concentration: AU4 + AU7.
    Concentration,
    /// Pain: AU4 + AU6 + AU7 + AU9 + AU10 + AU43.
    Pain,
    /// Thinking: AU4 + AU14 + AU64.
    Thinking,
}

impl ExpressionPreset {
    /// Get all available expression presets.
    pub fn all() -> &'static [ExpressionPreset] {
        &[
            ExpressionPreset::Neutral,
            ExpressionPreset::Joy,
            ExpressionPreset::Sadness,
            ExpressionPreset::Anger,
            ExpressionPreset::Fear,
            ExpressionPreset::Disgust,
            ExpressionPreset::Surprise,
            ExpressionPreset::Contempt,
            ExpressionPreset::Confusion,
            ExpressionPreset::Concentration,
            ExpressionPreset::Pain,
            ExpressionPreset::Thinking,
        ]
    }

    /// Get the action units and intensities for this preset.
    pub fn action_units(&self) -> Vec<(ActionUnit, f32)> {
        match self {
            ExpressionPreset::Neutral => vec![],
            ExpressionPreset::Joy => vec![
                (ActionUnit::AU6, INTENSITY_C),
                (ActionUnit::AU12, INTENSITY_C),
            ],
            ExpressionPreset::Sadness => vec![
                (ActionUnit::AU1, INTENSITY_B),
                (ActionUnit::AU4, INTENSITY_B),
                (ActionUnit::AU15, INTENSITY_B),
            ],
            ExpressionPreset::Anger => vec![
                (ActionUnit::AU4, INTENSITY_C),
                (ActionUnit::AU5, INTENSITY_B),
                (ActionUnit::AU7, INTENSITY_B),
                (ActionUnit::AU23, INTENSITY_B),
            ],
            ExpressionPreset::Fear => vec![
                (ActionUnit::AU1, INTENSITY_C),
                (ActionUnit::AU2, INTENSITY_C),
                (ActionUnit::AU4, INTENSITY_B),
                (ActionUnit::AU5, INTENSITY_C),
                (ActionUnit::AU20, INTENSITY_B),
                (ActionUnit::AU26, INTENSITY_B),
            ],
            ExpressionPreset::Disgust => vec![
                (ActionUnit::AU9, INTENSITY_C),
                (ActionUnit::AU15, INTENSITY_B),
                (ActionUnit::AU17, INTENSITY_A),
            ],
            ExpressionPreset::Surprise => vec![
                (ActionUnit::AU1, INTENSITY_C),
                (ActionUnit::AU2, INTENSITY_C),
                (ActionUnit::AU5, INTENSITY_C),
                (ActionUnit::AU26, INTENSITY_C),
            ],
            ExpressionPreset::Contempt => vec![
                (ActionUnit::AU12R, INTENSITY_B), // Unilateral
            ],
            ExpressionPreset::Confusion => vec![
                (ActionUnit::AU4, INTENSITY_B),
                (ActionUnit::AU1, INTENSITY_B),
                (ActionUnit::AU7, INTENSITY_A),
            ],
            ExpressionPreset::Concentration => vec![
                (ActionUnit::AU4, INTENSITY_B),
                (ActionUnit::AU7, INTENSITY_B),
            ],
            ExpressionPreset::Pain => vec![
                (ActionUnit::AU4, INTENSITY_C),
                (ActionUnit::AU6, INTENSITY_B),
                (ActionUnit::AU7, INTENSITY_C),
                (ActionUnit::AU9, INTENSITY_B),
                (ActionUnit::AU10, INTENSITY_B),
                (ActionUnit::AU43, INTENSITY_C),
            ],
            ExpressionPreset::Thinking => vec![
                (ActionUnit::AU4, INTENSITY_A),
                (ActionUnit::AU7, INTENSITY_A),
            ],
        }
    }

    /// Get a human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            ExpressionPreset::Neutral => "Neutral",
            ExpressionPreset::Joy => "Joy",
            ExpressionPreset::Sadness => "Sadness",
            ExpressionPreset::Anger => "Anger",
            ExpressionPreset::Fear => "Fear",
            ExpressionPreset::Disgust => "Disgust",
            ExpressionPreset::Surprise => "Surprise",
            ExpressionPreset::Contempt => "Contempt",
            ExpressionPreset::Confusion => "Confusion",
            ExpressionPreset::Concentration => "Concentration",
            ExpressionPreset::Pain => "Pain",
            ExpressionPreset::Thinking => "Thinking",
        }
    }
}

impl Default for ExpressionPreset {
    fn default() -> Self {
        ExpressionPreset::Neutral
    }
}

impl fmt::Display for ExpressionPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// AUCombinationType
// ---------------------------------------------------------------------------

/// How action units combine with each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AUCombinationType {
    /// AUs can be added together (default).
    Additive,
    /// AUs are mutually exclusive (conflict).
    Exclusive,
    /// One AU modifies another.
    Modifier,
}

impl Default for AUCombinationType {
    fn default() -> Self {
        AUCombinationType::Additive
    }
}

// ---------------------------------------------------------------------------
// AUConflict
// ---------------------------------------------------------------------------

/// Describes a conflict between two action units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AUConflict {
    /// First conflicting AU.
    pub au_a: ActionUnit,
    /// Second conflicting AU.
    pub au_b: ActionUnit,
    /// Type of conflict.
    pub conflict_type: AUCombinationType,
    /// Description of the conflict.
    pub description: String,
}

impl AUConflict {
    /// Create a new conflict.
    pub fn new(
        au_a: ActionUnit,
        au_b: ActionUnit,
        conflict_type: AUCombinationType,
        description: impl Into<String>,
    ) -> Self {
        Self {
            au_a,
            au_b,
            conflict_type,
            description: description.into(),
        }
    }

    /// Check if this conflict involves the given AU.
    pub fn involves(&self, au: ActionUnit) -> bool {
        self.au_a == au || self.au_b == au
    }

    /// Check if this conflict is between the two given AUs.
    pub fn is_between(&self, au_a: ActionUnit, au_b: ActionUnit) -> bool {
        (self.au_a == au_a && self.au_b == au_b) || (self.au_a == au_b && self.au_b == au_a)
    }
}

// ---------------------------------------------------------------------------
// DecayCurve
// ---------------------------------------------------------------------------

/// Decay curve types for natural AU falloff.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecayCurve {
    /// Linear decay.
    Linear,
    /// Exponential decay (faster at start).
    Exponential,
    /// Smooth step (ease in/out).
    SmoothStep,
    /// Elastic bounce effect.
    Elastic,
}

impl DecayCurve {
    /// Apply the decay curve to a value.
    ///
    /// `t` is normalized time (0.0 at start, 1.0 at end).
    /// Returns the intensity multiplier (1.0 at start, 0.0 at end).
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            DecayCurve::Linear => 1.0 - t,
            DecayCurve::Exponential => (1.0 - t).powi(2),
            DecayCurve::SmoothStep => {
                let t2 = t * t * (3.0 - 2.0 * t);
                1.0 - t2
            }
            DecayCurve::Elastic => {
                if t >= 1.0 {
                    0.0
                } else {
                    let p = 0.3;
                    let s = p / 4.0;
                    let inv_t = 1.0 - t;
                    inv_t.powf(2.0)
                        * ((inv_t * 10.0 - s) * std::f32::consts::TAU / p).sin()
                        * 0.5
                        + 0.5
                }
            }
        }
    }
}

impl Default for DecayCurve {
    fn default() -> Self {
        DecayCurve::SmoothStep
    }
}

// ---------------------------------------------------------------------------
// TransitionState
// ---------------------------------------------------------------------------

/// State for smooth transitions between expressions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TransitionState {
    /// Source weights (before transition).
    pub source: HashMap<ActionUnit, f32>,
    /// Target weights (after transition).
    pub target: HashMap<ActionUnit, f32>,
    /// Transition duration in seconds.
    pub duration: f32,
    /// Time elapsed in transition.
    pub elapsed: f32,
    /// Curve used for interpolation.
    pub curve: DecayCurve,
    /// Whether transition is active.
    pub active: bool,
}

impl TransitionState {
    /// Create a new transition.
    pub fn new(
        source: HashMap<ActionUnit, f32>,
        target: HashMap<ActionUnit, f32>,
        duration: f32,
    ) -> Self {
        Self {
            source,
            target,
            duration,
            elapsed: 0.0,
            curve: DecayCurve::SmoothStep,
            active: true,
        }
    }

    /// Builder: set curve.
    pub fn with_curve(mut self, curve: DecayCurve) -> Self {
        self.curve = curve;
        self
    }

    /// Get the normalized progress (0.0 - 1.0).
    #[inline]
    pub fn progress(&self) -> f32 {
        if self.duration <= 0.0 {
            1.0
        } else {
            (self.elapsed / self.duration).clamp(0.0, 1.0)
        }
    }

    /// Check if the transition is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }

    /// Update the transition with delta time.
    pub fn update(&mut self, dt: f32) {
        self.elapsed += dt;
        if self.elapsed >= self.duration {
            self.active = false;
        }
    }

    /// Get the current interpolated weights.
    pub fn current_weights(&self) -> HashMap<ActionUnit, f32> {
        let t = self.progress();
        // Use smooth step for the transition curve
        let t_curved = match self.curve {
            DecayCurve::Linear => t,
            DecayCurve::SmoothStep => t * t * (3.0 - 2.0 * t),
            DecayCurve::Exponential => 1.0 - (1.0 - t).powi(2),
            DecayCurve::Elastic => {
                if t >= 1.0 {
                    1.0
                } else {
                    let p = 0.3;
                    let s = p / 4.0;
                    t.powf(2.0) * ((t * 10.0 - s) * std::f32::consts::TAU / p).sin() * 0.5 + 0.5
                }
            }
        };

        let mut result = HashMap::new();

        // Get all unique AUs
        let all_aus: std::collections::HashSet<_> = self
            .source
            .keys()
            .chain(self.target.keys())
            .copied()
            .collect();

        for au in all_aus {
            let source_val = self.source.get(&au).copied().unwrap_or(0.0);
            let target_val = self.target.get(&au).copied().unwrap_or(0.0);
            let blended = source_val + (target_val - source_val) * t_curved;
            if blended > AU_WEIGHT_THRESHOLD {
                result.insert(au, blended);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// FACSController
// ---------------------------------------------------------------------------

/// Main controller for FACS-based facial animation.
///
/// Manages action unit weights, handles combinations and conflicts,
/// and outputs blend shape weights for rendering.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FACSController {
    /// Current action unit weights.
    weights: HashMap<ActionUnit, ActionUnitWeight>,
    /// Active transition state (if any).
    transition: Option<TransitionState>,
    /// Global intensity multiplier.
    pub global_intensity: f32,
    /// Whether to enforce AU conflict rules.
    pub enforce_conflicts: bool,
    /// Custom AU-to-blend-shape mapping overrides.
    pub blend_shape_mappings: HashMap<ActionUnit, String>,
}

impl FACSController {
    /// Create a new FACS controller.
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            transition: None,
            global_intensity: 1.0,
            enforce_conflicts: true,
            blend_shape_mappings: HashMap::new(),
        }
    }

    /// Set the intensity for an action unit.
    pub fn set_au(&mut self, au: ActionUnit, intensity: f32) {
        let clamped = intensity.clamp(INTENSITY_MIN, INTENSITY_MAX);
        if clamped > AU_WEIGHT_THRESHOLD {
            self.weights
                .entry(au)
                .and_modify(|w| w.intensity = clamped)
                .or_insert_with(|| ActionUnitWeight::new(au, clamped));
        } else {
            self.weights.remove(&au);
        }
    }

    /// Set an action unit with full configuration.
    pub fn set_au_weight(&mut self, weight: ActionUnitWeight) {
        if weight.intensity > AU_WEIGHT_THRESHOLD {
            self.weights.insert(weight.action_unit, weight);
        } else {
            self.weights.remove(&weight.action_unit);
        }
    }

    /// Get the intensity for an action unit.
    pub fn get_au(&self, au: ActionUnit) -> f32 {
        self.weights.get(&au).map_or(0.0, |w| w.intensity)
    }

    /// Get the full weight info for an action unit.
    pub fn get_au_weight(&self, au: ActionUnit) -> Option<&ActionUnitWeight> {
        self.weights.get(&au)
    }

    /// Clear a specific action unit.
    pub fn clear_au(&mut self, au: ActionUnit) {
        self.weights.remove(&au);
    }

    /// Reset all action units to zero.
    pub fn reset(&mut self) {
        self.weights.clear();
        self.transition = None;
    }

    /// Get all active action units.
    pub fn active_aus(&self) -> Vec<ActionUnit> {
        self.weights
            .iter()
            .filter(|(_, w)| w.is_active())
            .map(|(au, _)| *au)
            .collect()
    }

    /// Get the number of active action units.
    pub fn active_count(&self) -> usize {
        self.weights.values().filter(|w| w.is_active()).count()
    }

    /// Check if any action units are active.
    pub fn has_active_aus(&self) -> bool {
        self.weights.values().any(|w| w.is_active())
    }

    /// Apply an expression preset.
    pub fn apply_preset(&mut self, preset: ExpressionPreset) {
        // Clear existing
        self.reset();

        // Apply preset AUs
        for (au, intensity) in preset.action_units() {
            self.set_au(au, intensity);
        }
    }

    /// Apply an expression preset with smooth transition.
    pub fn transition_to_preset(&mut self, preset: ExpressionPreset, duration: f32) {
        let source: HashMap<_, _> = self
            .weights
            .iter()
            .map(|(au, w)| (*au, w.intensity))
            .collect();

        let target: HashMap<_, _> = preset
            .action_units()
            .into_iter()
            .map(|(au, intensity)| (au, intensity))
            .collect();

        self.transition = Some(TransitionState::new(source, target, duration));
    }

    /// Update the controller with delta time.
    pub fn update(&mut self, dt: f32) {
        // Update transition
        if let Some(ref mut transition) = self.transition {
            transition.update(dt);

            if transition.is_complete() {
                // Apply final target weights
                let final_weights = transition.target.clone();
                self.weights.clear();
                for (au, intensity) in final_weights {
                    self.set_au(au, intensity);
                }
                self.transition = None;
            }
        }

        // Update individual AU weights (decay, duration)
        let expired: Vec<_> = self
            .weights
            .iter_mut()
            .filter_map(|(au, w)| {
                w.update(dt);
                if w.is_expired() || !w.is_active() {
                    Some(*au)
                } else {
                    None
                }
            })
            .collect();

        for au in expired {
            self.weights.remove(&au);
        }
    }

    /// Get current effective weights (accounting for transitions).
    pub fn effective_weights(&self) -> HashMap<ActionUnit, f32> {
        if let Some(ref transition) = self.transition {
            if transition.active {
                return transition.current_weights();
            }
        }

        self.weights
            .iter()
            .filter(|(_, w)| w.is_active())
            .map(|(au, w)| (*au, w.intensity * self.global_intensity))
            .collect()
    }

    /// Convert current state to blend shape weights.
    ///
    /// Returns a map of blend shape names to weights (0.0 - 1.0).
    pub fn to_blend_weights(&self) -> HashMap<String, f32> {
        let effective = self.effective_weights();
        let mut result = HashMap::new();

        for (au, intensity) in effective {
            let blend_name = self
                .blend_shape_mappings
                .get(&au)
                .map(|s| s.as_str())
                .unwrap_or_else(|| au.blend_shape_name());

            // Normalize intensity to 0.0 - 1.0 range
            let weight = (intensity / INTENSITY_MAX).clamp(0.0, 1.0);

            result
                .entry(blend_name.to_string())
                .and_modify(|w: &mut f32| *w = (*w + weight).min(1.0))
                .or_insert(weight);
        }

        result
    }

    /// Apply weights to a BlendShapeSet.
    pub fn apply_to_blend_shapes(&self, shapes: &mut BlendShapeSet) {
        let blend_weights = self.to_blend_weights();

        for (name, weight) in blend_weights {
            shapes.set_weight_by_name(&name, weight);
        }
    }

    /// Detect conflicts between currently active AUs.
    pub fn detect_conflicts(&self) -> Vec<AUConflict> {
        let mut conflicts = Vec::new();
        let active: Vec<_> = self.active_aus();

        // Check known conflict pairs
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                if let Some(conflict) = Self::check_conflict(active[i], active[j]) {
                    conflicts.push(conflict);
                }
            }
        }

        conflicts
    }

    /// Check if two AUs conflict.
    fn check_conflict(au_a: ActionUnit, au_b: ActionUnit) -> Option<AUConflict> {
        // Known exclusive pairs
        let exclusive_pairs = [
            // Can't smile and frown at same time
            (ActionUnit::AU12, ActionUnit::AU15),
            (ActionUnit::AU12L, ActionUnit::AU15L),
            (ActionUnit::AU12R, ActionUnit::AU15R),
            // Can't raise and lower brow at same time
            (ActionUnit::AU1, ActionUnit::AU4),
            (ActionUnit::AU1L, ActionUnit::AU4L),
            (ActionUnit::AU1R, ActionUnit::AU4R),
            (ActionUnit::AU2, ActionUnit::AU4),
            (ActionUnit::AU2L, ActionUnit::AU4L),
            (ActionUnit::AU2R, ActionUnit::AU4R),
            // Can't have eyes wide open and closed
            (ActionUnit::AU5, ActionUnit::AU43),
            (ActionUnit::AU5L, ActionUnit::AU43L),
            (ActionUnit::AU5R, ActionUnit::AU43R),
            (ActionUnit::AU5, ActionUnit::AU45),
            (ActionUnit::AU5L, ActionUnit::AU45L),
            (ActionUnit::AU5R, ActionUnit::AU45R),
        ];

        for (a, b) in exclusive_pairs {
            if (au_a == a && au_b == b) || (au_a == b && au_b == a) {
                return Some(AUConflict::new(
                    au_a,
                    au_b,
                    AUCombinationType::Exclusive,
                    format!(
                        "{} and {} are anatomically exclusive",
                        au_a.name(),
                        au_b.name()
                    ),
                ));
            }
        }

        None
    }

    /// Resolve conflicts by reducing intensity of conflicting AUs.
    pub fn resolve_conflicts(&mut self) {
        if !self.enforce_conflicts {
            return;
        }

        let conflicts = self.detect_conflicts();

        for conflict in conflicts {
            if conflict.conflict_type == AUCombinationType::Exclusive {
                // Keep the stronger one, reduce the weaker
                let intensity_a = self.get_au(conflict.au_a);
                let intensity_b = self.get_au(conflict.au_b);

                if intensity_a >= intensity_b {
                    self.clear_au(conflict.au_b);
                } else {
                    self.clear_au(conflict.au_a);
                }
            }
        }
    }

    /// Add additive AU with intensity stacking limits.
    pub fn add_au(&mut self, au: ActionUnit, intensity: f32) {
        let current = self.get_au(au);
        let new_intensity = (current + intensity).min(MAX_STACKED_INTENSITY);
        self.set_au(au, new_intensity);
    }

    /// Set a custom blend shape mapping for an AU.
    pub fn set_blend_mapping(&mut self, au: ActionUnit, blend_name: impl Into<String>) {
        self.blend_shape_mappings.insert(au, blend_name.into());
    }

    /// Create a FACS notation string for current expression.
    ///
    /// Example: "AU1B + AU12C + AU6C" (sadness/joy mix)
    pub fn to_facs_notation(&self) -> String {
        let mut parts: Vec<_> = self
            .weights
            .iter()
            .filter(|(_, w)| w.is_active())
            .map(|(au, w)| format!("{}{}", au.code(), w.facs_level()))
            .collect();

        parts.sort();
        parts.join(" + ")
    }

    /// Parse a FACS notation string and apply it.
    ///
    /// Format: "AU1B + AU12C" or "AU1B+AU12C"
    pub fn from_facs_notation(&mut self, notation: &str) -> Result<(), String> {
        self.reset();

        for part in notation.split('+') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            // Parse AU code and optional intensity
            let (code, level) = if part.ends_with(|c: char| c.is_ascii_uppercase()) {
                let level_char = part.chars().last().unwrap();
                let code = &part[..part.len() - 1];
                let level = match level_char {
                    'A' => INTENSITY_A,
                    'B' => INTENSITY_B,
                    'C' => INTENSITY_C,
                    'D' => INTENSITY_D,
                    'E' => INTENSITY_E,
                    _ => INTENSITY_C, // Default to C
                };
                (code, level)
            } else {
                (part, INTENSITY_C) // Default intensity
            };

            // Find matching AU
            let au = ActionUnit::all()
                .iter()
                .find(|au| au.code() == code)
                .copied()
                .ok_or_else(|| format!("Unknown action unit: {}", code))?;

            self.set_au(au, level);
        }

        Ok(())
    }

    /// Get the current expression as a compact serializable state.
    pub fn snapshot(&self) -> HashMap<ActionUnit, f32> {
        self.weights
            .iter()
            .filter(|(_, w)| w.is_active())
            .map(|(au, w)| (*au, w.intensity))
            .collect()
    }

    /// Restore from a snapshot.
    pub fn restore(&mut self, snapshot: &HashMap<ActionUnit, f32>) {
        self.reset();
        for (au, intensity) in snapshot {
            self.set_au(*au, *intensity);
        }
    }

    /// Blend two expressions together.
    pub fn blend_with(&mut self, other: &FACSController, blend_factor: f32) {
        let factor = blend_factor.clamp(0.0, 1.0);

        // Get all AUs from both controllers
        let all_aus: std::collections::HashSet<_> = self
            .weights
            .keys()
            .chain(other.weights.keys())
            .copied()
            .collect();

        for au in all_aus {
            let self_intensity = self.get_au(au);
            let other_intensity = other.get_au(au);
            let blended = self_intensity * (1.0 - factor) + other_intensity * factor;
            self.set_au(au, blended);
        }
    }
}

// ---------------------------------------------------------------------------
// ExpressionBuilder
// ---------------------------------------------------------------------------

/// Builder for creating custom expressions from action units.
#[derive(Clone, Debug, Default)]
pub struct ExpressionBuilder {
    aus: Vec<(ActionUnit, f32, f32)>, // AU, intensity, asymmetry
    name: Option<String>,
}

impl ExpressionBuilder {
    /// Create a new expression builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a name for this expression.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add an action unit with intensity.
    pub fn au(mut self, au: ActionUnit, intensity: f32) -> Self {
        self.aus.push((au, intensity.clamp(0.0, INTENSITY_MAX), 0.0));
        self
    }

    /// Add an action unit with intensity and asymmetry.
    pub fn au_asymmetric(mut self, au: ActionUnit, intensity: f32, asymmetry: f32) -> Self {
        self.aus.push((
            au,
            intensity.clamp(0.0, INTENSITY_MAX),
            asymmetry.clamp(-1.0, 1.0),
        ));
        self
    }

    /// Add multiple action units at the same intensity.
    pub fn aus(mut self, aus: &[ActionUnit], intensity: f32) -> Self {
        for au in aus {
            self.aus.push((*au, intensity.clamp(0.0, INTENSITY_MAX), 0.0));
        }
        self
    }

    /// Build and apply to a controller.
    pub fn apply_to(self, controller: &mut FACSController) {
        for (au, intensity, asymmetry) in self.aus {
            let weight = ActionUnitWeight::new(au, intensity).with_asymmetry(asymmetry);
            controller.set_au_weight(weight);
        }
    }

    /// Build into a controller.
    pub fn build(self) -> FACSController {
        let mut controller = FACSController::new();
        self.apply_to(&mut controller);
        controller
    }

    /// Get the expression name if set.
    pub fn expression_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ActionUnit Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_action_unit_all_base() {
        let all = ActionUnit::all_base();
        assert_eq!(all.len(), 18);
    }

    #[test]
    fn test_action_unit_all() {
        let all = ActionUnit::all();
        assert_eq!(all.len(), 44);
    }

    #[test]
    fn test_action_unit_code() {
        assert_eq!(ActionUnit::AU1.code(), "AU1");
        assert_eq!(ActionUnit::AU12L.code(), "AU12L");
        assert_eq!(ActionUnit::AU45R.code(), "AU45R");
    }

    #[test]
    fn test_action_unit_name() {
        assert_eq!(ActionUnit::AU1.name(), "Inner Brow Raiser");
        assert_eq!(ActionUnit::AU12.name(), "Lip Corner Puller");
        assert_eq!(ActionUnit::AU45.name(), "Blink");
    }

    #[test]
    fn test_action_unit_face_region() {
        assert_eq!(ActionUnit::AU1.face_region(), FaceRegion::LeftBrow);
        assert_eq!(ActionUnit::AU5L.face_region(), FaceRegion::LeftEye);
        assert_eq!(ActionUnit::AU5R.face_region(), FaceRegion::RightEye);
        assert_eq!(ActionUnit::AU12.face_region(), FaceRegion::Mouth);
        assert_eq!(ActionUnit::AU9.face_region(), FaceRegion::Nose);
    }

    #[test]
    fn test_action_unit_bilateral() {
        assert!(ActionUnit::AU1.is_bilateral());
        assert!(ActionUnit::AU12.is_bilateral());
        assert!(!ActionUnit::AU1L.is_bilateral());
        assert!(!ActionUnit::AU17.is_bilateral());
    }

    #[test]
    fn test_action_unit_left_right() {
        assert!(ActionUnit::AU1L.is_left());
        assert!(!ActionUnit::AU1R.is_left());
        assert!(ActionUnit::AU1R.is_right());
        assert!(!ActionUnit::AU1L.is_right());
    }

    #[test]
    fn test_action_unit_variants() {
        assert_eq!(ActionUnit::AU1.left_variant(), Some(ActionUnit::AU1L));
        assert_eq!(ActionUnit::AU1.right_variant(), Some(ActionUnit::AU1R));
        assert_eq!(ActionUnit::AU17.left_variant(), None);
    }

    #[test]
    fn test_action_unit_base() {
        assert_eq!(ActionUnit::AU1L.base(), ActionUnit::AU1);
        assert_eq!(ActionUnit::AU1R.base(), ActionUnit::AU1);
        assert_eq!(ActionUnit::AU1.base(), ActionUnit::AU1);
        assert_eq!(ActionUnit::AU17.base(), ActionUnit::AU17);
    }

    #[test]
    fn test_action_unit_blend_shape_name() {
        assert_eq!(ActionUnit::AU12.blend_shape_name(), "mouthSmile");
        assert_eq!(ActionUnit::AU12L.blend_shape_name(), "mouthSmile_L");
        assert_eq!(ActionUnit::AU45.blend_shape_name(), "blink");
    }

    #[test]
    fn test_action_unit_display() {
        let display = format!("{}", ActionUnit::AU12);
        assert!(display.contains("AU12"));
        assert!(display.contains("Lip Corner Puller"));
    }

    #[test]
    fn test_action_unit_default() {
        let au = ActionUnit::default();
        assert_eq!(au, ActionUnit::AU12);
    }

    // -----------------------------------------------------------------------
    // ActionUnitWeight Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_au_weight_new() {
        let w = ActionUnitWeight::new(ActionUnit::AU12, 3.0);
        assert_eq!(w.action_unit, ActionUnit::AU12);
        assert_eq!(w.intensity, 3.0);
        assert_eq!(w.asymmetry, 0.0);
    }

    #[test]
    fn test_au_weight_clamping() {
        let w1 = ActionUnitWeight::new(ActionUnit::AU1, -1.0);
        assert_eq!(w1.intensity, 0.0);

        let w2 = ActionUnitWeight::new(ActionUnit::AU1, 10.0);
        assert_eq!(w2.intensity, 5.0);
    }

    #[test]
    fn test_au_weight_builder() {
        let w = ActionUnitWeight::new(ActionUnit::AU12, 3.0)
            .with_asymmetry(0.5)
            .with_duration(2.0)
            .with_decay(0.5);

        assert_eq!(w.asymmetry, 0.5);
        assert_eq!(w.duration, 2.0);
        assert_eq!(w.decay_rate, 0.5);
    }

    #[test]
    fn test_au_weight_asymmetry_clamping() {
        let w1 = ActionUnitWeight::new(ActionUnit::AU1, 3.0).with_asymmetry(-2.0);
        assert_eq!(w1.asymmetry, -1.0);

        let w2 = ActionUnitWeight::new(ActionUnit::AU1, 3.0).with_asymmetry(2.0);
        assert_eq!(w2.asymmetry, 1.0);
    }

    #[test]
    fn test_au_weight_is_active() {
        let active = ActionUnitWeight::new(ActionUnit::AU1, 1.0);
        assert!(active.is_active());

        let inactive = ActionUnitWeight::new(ActionUnit::AU1, 0.0);
        assert!(!inactive.is_active());
    }

    #[test]
    fn test_au_weight_is_expired() {
        let mut w = ActionUnitWeight::new(ActionUnit::AU1, 1.0).with_duration(1.0);

        assert!(!w.is_expired());

        w.elapsed = 1.0;
        assert!(w.is_expired());
    }

    #[test]
    fn test_au_weight_update() {
        let mut w = ActionUnitWeight::new(ActionUnit::AU1, 2.0).with_decay(1.0);

        w.update(0.5);
        assert!((w.intensity - 1.5).abs() < 0.01);
        assert!((w.elapsed - 0.5).abs() < 0.01);

        w.update(2.0);
        assert!(w.intensity <= 0.0);
    }

    #[test]
    fn test_au_weight_left_right_intensity() {
        // Symmetric
        let sym = ActionUnitWeight::new(ActionUnit::AU12, 5.0);
        assert!((sym.left_intensity() - 1.0).abs() < 0.01);
        assert!((sym.right_intensity() - 1.0).abs() < 0.01);

        // Left-biased
        let left = ActionUnitWeight::new(ActionUnit::AU12, 5.0).with_asymmetry(-0.5);
        assert!(left.left_intensity() > left.right_intensity());

        // Right-biased
        let right = ActionUnitWeight::new(ActionUnit::AU12, 5.0).with_asymmetry(0.5);
        assert!(right.right_intensity() > right.left_intensity());
    }

    #[test]
    fn test_au_weight_normalized_intensity() {
        let w = ActionUnitWeight::new(ActionUnit::AU1, 2.5);
        assert!((w.normalized_intensity() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_au_weight_facs_level() {
        assert_eq!(
            ActionUnitWeight::new(ActionUnit::AU1, 0.0).facs_level(),
            FACSIntensityLevel::None
        );
        assert_eq!(
            ActionUnitWeight::new(ActionUnit::AU1, 1.0).facs_level(),
            FACSIntensityLevel::A
        );
        assert_eq!(
            ActionUnitWeight::new(ActionUnit::AU1, 2.0).facs_level(),
            FACSIntensityLevel::B
        );
        assert_eq!(
            ActionUnitWeight::new(ActionUnit::AU1, 3.0).facs_level(),
            FACSIntensityLevel::C
        );
        assert_eq!(
            ActionUnitWeight::new(ActionUnit::AU1, 4.0).facs_level(),
            FACSIntensityLevel::D
        );
        assert_eq!(
            ActionUnitWeight::new(ActionUnit::AU1, 5.0).facs_level(),
            FACSIntensityLevel::E
        );
    }

    // -----------------------------------------------------------------------
    // FACSIntensityLevel Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_facs_level_from_value() {
        assert_eq!(FACSIntensityLevel::from_value(0.0), FACSIntensityLevel::None);
        assert_eq!(FACSIntensityLevel::from_value(0.4), FACSIntensityLevel::None);
        assert_eq!(FACSIntensityLevel::from_value(0.5), FACSIntensityLevel::A);
        assert_eq!(FACSIntensityLevel::from_value(1.0), FACSIntensityLevel::A);
        assert_eq!(FACSIntensityLevel::from_value(1.5), FACSIntensityLevel::B);
        assert_eq!(FACSIntensityLevel::from_value(2.5), FACSIntensityLevel::C);
        assert_eq!(FACSIntensityLevel::from_value(3.5), FACSIntensityLevel::D);
        assert_eq!(FACSIntensityLevel::from_value(4.5), FACSIntensityLevel::E);
    }

    #[test]
    fn test_facs_level_to_value() {
        assert_eq!(FACSIntensityLevel::None.to_value(), 0.0);
        assert_eq!(FACSIntensityLevel::A.to_value(), 1.0);
        assert_eq!(FACSIntensityLevel::B.to_value(), 2.0);
        assert_eq!(FACSIntensityLevel::C.to_value(), 3.0);
        assert_eq!(FACSIntensityLevel::D.to_value(), 4.0);
        assert_eq!(FACSIntensityLevel::E.to_value(), 5.0);
    }

    #[test]
    fn test_facs_level_code() {
        assert_eq!(FACSIntensityLevel::None.code(), '-');
        assert_eq!(FACSIntensityLevel::A.code(), 'A');
        assert_eq!(FACSIntensityLevel::E.code(), 'E');
    }

    #[test]
    fn test_facs_level_display() {
        assert_eq!(format!("{}", FACSIntensityLevel::C), "C");
    }

    // -----------------------------------------------------------------------
    // ExpressionPreset Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_expression_preset_all() {
        assert_eq!(ExpressionPreset::all().len(), 12);
    }

    #[test]
    fn test_expression_preset_joy() {
        let aus = ExpressionPreset::Joy.action_units();
        assert_eq!(aus.len(), 2);
        assert!(aus.iter().any(|(au, _)| *au == ActionUnit::AU6));
        assert!(aus.iter().any(|(au, _)| *au == ActionUnit::AU12));
    }

    #[test]
    fn test_expression_preset_sadness() {
        let aus = ExpressionPreset::Sadness.action_units();
        assert_eq!(aus.len(), 3);
        assert!(aus.iter().any(|(au, _)| *au == ActionUnit::AU1));
        assert!(aus.iter().any(|(au, _)| *au == ActionUnit::AU4));
        assert!(aus.iter().any(|(au, _)| *au == ActionUnit::AU15));
    }

    #[test]
    fn test_expression_preset_neutral() {
        let aus = ExpressionPreset::Neutral.action_units();
        assert!(aus.is_empty());
    }

    #[test]
    fn test_expression_preset_name() {
        assert_eq!(ExpressionPreset::Joy.name(), "Joy");
        assert_eq!(ExpressionPreset::Anger.name(), "Anger");
    }

    #[test]
    fn test_expression_preset_contempt_unilateral() {
        let aus = ExpressionPreset::Contempt.action_units();
        assert!(aus.iter().any(|(au, _)| *au == ActionUnit::AU12R));
    }

    // -----------------------------------------------------------------------
    // DecayCurve Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decay_curve_linear() {
        assert!((DecayCurve::Linear.apply(0.0) - 1.0).abs() < 0.01);
        assert!((DecayCurve::Linear.apply(0.5) - 0.5).abs() < 0.01);
        assert!((DecayCurve::Linear.apply(1.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_decay_curve_exponential() {
        assert!((DecayCurve::Exponential.apply(0.0) - 1.0).abs() < 0.01);
        assert!((DecayCurve::Exponential.apply(1.0) - 0.0).abs() < 0.01);
        // Exponential decays faster at start
        assert!(DecayCurve::Exponential.apply(0.5) < 0.5);
    }

    #[test]
    fn test_decay_curve_smooth_step() {
        assert!((DecayCurve::SmoothStep.apply(0.0) - 1.0).abs() < 0.01);
        assert!((DecayCurve::SmoothStep.apply(1.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_decay_curve_clamping() {
        // Should clamp to valid range
        assert!((DecayCurve::Linear.apply(-1.0) - 1.0).abs() < 0.01);
        assert!((DecayCurve::Linear.apply(2.0) - 0.0).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // TransitionState Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_transition_state_new() {
        let source: HashMap<_, _> = vec![(ActionUnit::AU12, 3.0)].into_iter().collect();
        let target: HashMap<_, _> = vec![(ActionUnit::AU15, 2.0)].into_iter().collect();

        let transition = TransitionState::new(source, target, 0.5);

        assert!(transition.active);
        assert_eq!(transition.duration, 0.5);
        assert_eq!(transition.elapsed, 0.0);
    }

    #[test]
    fn test_transition_state_progress() {
        let mut transition =
            TransitionState::new(HashMap::new(), HashMap::new(), 1.0);

        assert_eq!(transition.progress(), 0.0);

        transition.elapsed = 0.5;
        assert!((transition.progress() - 0.5).abs() < 0.01);

        transition.elapsed = 1.0;
        assert!((transition.progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_transition_state_is_complete() {
        let mut transition =
            TransitionState::new(HashMap::new(), HashMap::new(), 1.0);

        assert!(!transition.is_complete());

        transition.elapsed = 1.0;
        assert!(transition.is_complete());
    }

    #[test]
    fn test_transition_state_update() {
        let mut transition =
            TransitionState::new(HashMap::new(), HashMap::new(), 1.0);

        transition.update(0.5);
        assert!((transition.elapsed - 0.5).abs() < 0.01);
        assert!(transition.active);

        transition.update(0.6);
        assert!(!transition.active);
    }

    #[test]
    fn test_transition_state_current_weights() {
        let source: HashMap<_, _> = vec![(ActionUnit::AU12, 0.0)].into_iter().collect();
        let target: HashMap<_, _> = vec![(ActionUnit::AU12, 5.0)].into_iter().collect();

        let mut transition = TransitionState::new(source, target, 1.0)
            .with_curve(DecayCurve::Linear);

        let start = transition.current_weights();
        assert!(start.get(&ActionUnit::AU12).copied().unwrap_or(0.0) < 0.1);

        transition.elapsed = 0.5;
        let mid = transition.current_weights();
        let mid_val = mid.get(&ActionUnit::AU12).copied().unwrap_or(0.0);
        assert!(mid_val > 2.0 && mid_val < 3.0);

        transition.elapsed = 1.0;
        let end = transition.current_weights();
        assert!((end.get(&ActionUnit::AU12).copied().unwrap_or(0.0) - 5.0).abs() < 0.1);
    }

    // -----------------------------------------------------------------------
    // FACSController Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_facs_controller_new() {
        let controller = FACSController::new();
        assert!(!controller.has_active_aus());
        assert_eq!(controller.active_count(), 0);
    }

    #[test]
    fn test_facs_controller_set_get_au() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 3.0);

        assert_eq!(controller.get_au(ActionUnit::AU12), 3.0);
        assert_eq!(controller.get_au(ActionUnit::AU1), 0.0);
    }

    #[test]
    fn test_facs_controller_set_au_weight() {
        let mut controller = FACSController::new();
        let weight = ActionUnitWeight::new(ActionUnit::AU12, 3.0).with_asymmetry(0.5);
        controller.set_au_weight(weight);

        let retrieved = controller.get_au_weight(ActionUnit::AU12).unwrap();
        assert_eq!(retrieved.asymmetry, 0.5);
    }

    #[test]
    fn test_facs_controller_clear_au() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 3.0);
        assert!(controller.has_active_aus());

        controller.clear_au(ActionUnit::AU12);
        assert!(!controller.has_active_aus());
    }

    #[test]
    fn test_facs_controller_reset() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 3.0);
        controller.set_au(ActionUnit::AU6, 2.0);

        controller.reset();
        assert!(!controller.has_active_aus());
    }

    #[test]
    fn test_facs_controller_active_aus() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 3.0);
        controller.set_au(ActionUnit::AU6, 2.0);
        controller.set_au(ActionUnit::AU1, 0.0);

        let active = controller.active_aus();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&ActionUnit::AU12));
        assert!(active.contains(&ActionUnit::AU6));
    }

    #[test]
    fn test_facs_controller_apply_preset() {
        let mut controller = FACSController::new();
        controller.apply_preset(ExpressionPreset::Joy);

        assert!(controller.get_au(ActionUnit::AU6) > 0.0);
        assert!(controller.get_au(ActionUnit::AU12) > 0.0);
    }

    #[test]
    fn test_facs_controller_transition_to_preset() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU15, 3.0); // Start with frown
        controller.transition_to_preset(ExpressionPreset::Joy, 0.5);

        assert!(controller.transition.is_some());
    }

    #[test]
    fn test_facs_controller_update() {
        let mut controller = FACSController::new();
        let weight = ActionUnitWeight::new(ActionUnit::AU12, 3.0)
            .with_decay(1.0)
            .with_duration(1.0);
        controller.set_au_weight(weight);

        controller.update(0.5);
        assert!(controller.get_au(ActionUnit::AU12) < 3.0);
        assert!(controller.get_au(ActionUnit::AU12) > 2.0);
    }

    #[test]
    fn test_facs_controller_update_expires() {
        let mut controller = FACSController::new();
        let weight = ActionUnitWeight::new(ActionUnit::AU12, 3.0).with_duration(0.5);
        controller.set_au_weight(weight);

        controller.update(1.0);
        assert!(!controller.has_active_aus());
    }

    #[test]
    fn test_facs_controller_effective_weights() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 3.0);
        controller.global_intensity = 0.5;

        let effective = controller.effective_weights();
        assert!((effective.get(&ActionUnit::AU12).copied().unwrap_or(0.0) - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_facs_controller_to_blend_weights() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 5.0);

        let blend_weights = controller.to_blend_weights();
        let smile_weight = blend_weights.get("mouthSmile").copied().unwrap_or(0.0);
        assert!((smile_weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_facs_controller_custom_blend_mapping() {
        let mut controller = FACSController::new();
        controller.set_blend_mapping(ActionUnit::AU12, "smile_custom");
        controller.set_au(ActionUnit::AU12, 5.0);

        let blend_weights = controller.to_blend_weights();
        assert!(blend_weights.contains_key("smile_custom"));
        assert!(!blend_weights.contains_key("mouthSmile"));
    }

    #[test]
    fn test_facs_controller_detect_conflicts() {
        let mut controller = FACSController::new();
        controller.enforce_conflicts = false; // Don't auto-resolve
        controller.set_au(ActionUnit::AU12, 3.0); // Smile
        controller.set_au(ActionUnit::AU15, 3.0); // Frown

        let conflicts = controller.detect_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert!(conflicts[0].involves(ActionUnit::AU12));
        assert!(conflicts[0].involves(ActionUnit::AU15));
    }

    #[test]
    fn test_facs_controller_resolve_conflicts() {
        let mut controller = FACSController::new();
        controller.enforce_conflicts = true;
        controller.set_au(ActionUnit::AU12, 3.0); // Stronger
        controller.set_au(ActionUnit::AU15, 2.0); // Weaker

        controller.resolve_conflicts();

        // Stronger AU should remain
        assert!(controller.get_au(ActionUnit::AU12) > 0.0);
        assert!(controller.get_au(ActionUnit::AU15) == 0.0);
    }

    #[test]
    fn test_facs_controller_add_au() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 2.0);
        controller.add_au(ActionUnit::AU12, 1.5);

        assert!((controller.get_au(ActionUnit::AU12) - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_facs_controller_add_au_max_stack() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 5.0);
        controller.add_au(ActionUnit::AU12, 5.0);

        // Should be capped at MAX_STACKED_INTENSITY
        assert!(controller.get_au(ActionUnit::AU12) <= MAX_STACKED_INTENSITY);
    }

    #[test]
    fn test_facs_controller_facs_notation() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU1, 2.0);
        controller.set_au(ActionUnit::AU12, 3.0);

        let notation = controller.to_facs_notation();
        assert!(notation.contains("AU1B"));
        assert!(notation.contains("AU12C"));
        assert!(notation.contains("+"));
    }

    #[test]
    fn test_facs_controller_from_facs_notation() {
        let mut controller = FACSController::new();
        controller.from_facs_notation("AU1B + AU12C").unwrap();

        assert!((controller.get_au(ActionUnit::AU1) - INTENSITY_B).abs() < 0.1);
        assert!((controller.get_au(ActionUnit::AU12) - INTENSITY_C).abs() < 0.1);
    }

    #[test]
    fn test_facs_controller_from_facs_notation_invalid() {
        let mut controller = FACSController::new();
        let result = controller.from_facs_notation("AU999");
        assert!(result.is_err());
    }

    #[test]
    fn test_facs_controller_snapshot_restore() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 3.0);
        controller.set_au(ActionUnit::AU6, 2.0);

        let snapshot = controller.snapshot();

        controller.reset();
        assert!(!controller.has_active_aus());

        controller.restore(&snapshot);
        assert!((controller.get_au(ActionUnit::AU12) - 3.0).abs() < 0.01);
        assert!((controller.get_au(ActionUnit::AU6) - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_facs_controller_blend_with() {
        let mut controller1 = FACSController::new();
        controller1.set_au(ActionUnit::AU12, 4.0);

        let mut controller2 = FACSController::new();
        controller2.set_au(ActionUnit::AU12, 2.0);
        controller2.set_au(ActionUnit::AU6, 3.0);

        controller1.blend_with(&controller2, 0.5);

        assert!((controller1.get_au(ActionUnit::AU12) - 3.0).abs() < 0.01); // 4 * 0.5 + 2 * 0.5
        assert!((controller1.get_au(ActionUnit::AU6) - 1.5).abs() < 0.01); // 0 * 0.5 + 3 * 0.5
    }

    // -----------------------------------------------------------------------
    // ExpressionBuilder Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_expression_builder() {
        let controller = ExpressionBuilder::new()
            .name("Custom Smile")
            .au(ActionUnit::AU12, 3.0)
            .au(ActionUnit::AU6, 2.0)
            .build();

        assert!(controller.get_au(ActionUnit::AU12) > 0.0);
        assert!(controller.get_au(ActionUnit::AU6) > 0.0);
    }

    #[test]
    fn test_expression_builder_asymmetric() {
        let controller = ExpressionBuilder::new()
            .au_asymmetric(ActionUnit::AU12, 3.0, 0.5)
            .build();

        let weight = controller.get_au_weight(ActionUnit::AU12).unwrap();
        assert_eq!(weight.asymmetry, 0.5);
    }

    #[test]
    fn test_expression_builder_multiple_aus() {
        let controller = ExpressionBuilder::new()
            .aus(&[ActionUnit::AU1, ActionUnit::AU2, ActionUnit::AU5], 2.0)
            .build();

        assert!(controller.get_au(ActionUnit::AU1) > 0.0);
        assert!(controller.get_au(ActionUnit::AU2) > 0.0);
        assert!(controller.get_au(ActionUnit::AU5) > 0.0);
    }

    #[test]
    fn test_expression_builder_apply_to() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU15, 2.0); // Pre-existing

        ExpressionBuilder::new()
            .au(ActionUnit::AU12, 3.0)
            .apply_to(&mut controller);

        assert!(controller.get_au(ActionUnit::AU12) > 0.0);
        assert!(controller.get_au(ActionUnit::AU15) > 0.0);
    }

    // -----------------------------------------------------------------------
    // AUConflict Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_au_conflict() {
        let conflict = AUConflict::new(
            ActionUnit::AU12,
            ActionUnit::AU15,
            AUCombinationType::Exclusive,
            "Smile and frown conflict",
        );

        assert!(conflict.involves(ActionUnit::AU12));
        assert!(conflict.involves(ActionUnit::AU15));
        assert!(!conflict.involves(ActionUnit::AU1));
    }

    #[test]
    fn test_au_conflict_is_between() {
        let conflict = AUConflict::new(
            ActionUnit::AU12,
            ActionUnit::AU15,
            AUCombinationType::Exclusive,
            "Test",
        );

        assert!(conflict.is_between(ActionUnit::AU12, ActionUnit::AU15));
        assert!(conflict.is_between(ActionUnit::AU15, ActionUnit::AU12));
        assert!(!conflict.is_between(ActionUnit::AU12, ActionUnit::AU1));
    }

    // -----------------------------------------------------------------------
    // Edge Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_intensity() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 0.0);
        assert!(!controller.has_active_aus());
    }

    #[test]
    fn test_max_intensity() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 10.0); // Over max

        assert!((controller.get_au(ActionUnit::AU12) - INTENSITY_MAX).abs() < 0.01);
    }

    #[test]
    fn test_negative_intensity() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, -5.0); // Negative

        assert!(controller.get_au(ActionUnit::AU12) >= 0.0);
    }

    #[test]
    fn test_all_presets_valid() {
        for preset in ExpressionPreset::all() {
            let aus = preset.action_units();
            for (au, intensity) in aus {
                assert!(intensity >= INTENSITY_MIN);
                assert!(intensity <= INTENSITY_MAX);
                // AU should be valid
                assert!(ActionUnit::all().contains(&au));
            }
        }
    }

    #[test]
    fn test_apply_to_blend_shapes() {
        let mut shapes = BlendShapeSet::new(100);
        shapes.add_target(
            crate::blend_shapes::BlendShapeTarget::new("mouthSmile")
                .with_region(FaceRegion::Mouth),
        );

        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 5.0);
        controller.apply_to_blend_shapes(&mut shapes);

        // Weight should be set
        assert!(shapes.get_weight_by_name("mouthSmile").unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn test_transition_completes() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU15, 3.0);
        controller.transition_to_preset(ExpressionPreset::Joy, 0.5);

        // Run transition to completion
        controller.update(0.6);

        // Should have joy AUs now
        assert!(controller.get_au(ActionUnit::AU6) > 0.0);
        assert!(controller.get_au(ActionUnit::AU12) > 0.0);
        // Frown should be gone
        assert!(controller.get_au(ActionUnit::AU15) == 0.0);
    }

    #[test]
    fn test_default_impls() {
        let _au = ActionUnit::default();
        let _weight = ActionUnitWeight::default();
        let _level = FACSIntensityLevel::default();
        let _preset = ExpressionPreset::default();
        let _curve = DecayCurve::default();
        let _combination = AUCombinationType::default();
        let _transition = TransitionState::default();
        let _controller = FACSController::default();
        let _builder = ExpressionBuilder::default();
    }

    #[test]
    fn test_serialization() {
        let mut controller = FACSController::new();
        controller.set_au(ActionUnit::AU12, 3.0);
        controller.set_au(ActionUnit::AU6, 2.0);

        let serialized = serde_json::to_string(&controller).unwrap();
        let deserialized: FACSController = serde_json::from_str(&serialized).unwrap();

        assert!((deserialized.get_au(ActionUnit::AU12) - 3.0).abs() < 0.01);
        assert!((deserialized.get_au(ActionUnit::AU6) - 2.0).abs() < 0.01);
    }
}
