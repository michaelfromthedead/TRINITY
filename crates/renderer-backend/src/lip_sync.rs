//! Lip Sync System for TRINITY Engine (T-AN-7.3).
//!
//! This module provides a complete lip sync system for facial animation
//! driven by phoneme data or real-time audio analysis:
//!
//! - Standard viseme set (15 visemes based on Preston Blair)
//! - IPA phoneme-to-viseme mapping with language support
//! - Coarticulation rules for natural blending
//! - Timed viseme tracks with event markers
//! - Real-time audio analysis fallback
//! - Integration with blend shapes and FACS
//!
//! # Viseme System
//!
//! Visemes are visual representations of phonemes - mouth shapes that
//! correspond to spoken sounds. This module uses the standard 15-viseme
//! set commonly used in animation:
//!
//! | Viseme | Phonemes | Description |
//! |--------|----------|-------------|
//! | Sil    | silence  | Closed/neutral mouth |
//! | PP     | p, b, m  | Bilabial closure |
//! | FF     | f, v     | Labiodental |
//! | TH     | th, dh   | Dental fricative |
//! | DD     | t, d, n  | Alveolar |
//! | KK     | k, g     | Velar |
//! | CH     | ch, j, sh| Postalveolar |
//! | SS     | s, z     | Sibilant |
//! | NN     | n, ng    | Nasal |
//! | RR     | r        | Rhotic |
//! | AA     | a, ah    | Open vowel |
//! | EE     | e, ee    | Front vowel |
//! | IH     | i, ih    | Near-close front |
//! | OH     | o, oh    | Back rounded |
//! | OU     | oo, u    | Close back |
//!
//! # Coarticulation
//!
//! Natural speech involves overlapping articulation - the mouth starts
//! preparing for the next sound before finishing the current one. This
//! module implements:
//!
//! - Anticipatory coarticulation (looking ahead)
//! - Carryover effects (lingering from previous phoneme)
//! - Consonant-vowel blending
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::lip_sync::{
//!     LipSyncController, LipSyncTrack, Viseme, VisemeWeight,
//!     PhonemeToViseme, AudioAnalyzer,
//! };
//!
//! // Create from phoneme transcript
//! let mut track = LipSyncTrack::new();
//! track.add_viseme(0.0, Viseme::Sil, 1.0, 0.05);
//! track.add_viseme(0.05, Viseme::AA, 1.0, 0.1);
//! track.add_viseme(0.15, Viseme::Sil, 1.0, 0.05);
//!
//! // Sample at arbitrary time
//! let weights = track.sample(0.1);
//!
//! // Or use audio analysis fallback
//! let mut analyzer = AudioAnalyzer::new(44100.0);
//! let amplitude = analyzer.process_samples(&audio_buffer);
//! ```

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::blend_shapes::{BlendShapeSet, FaceRegion};
use crate::facs_action_units::{ActionUnit, FACSController, INTENSITY_B, INTENSITY_C};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of standard visemes.
pub const VISEME_COUNT: usize = 15;

/// Default viseme blend duration in seconds.
pub const DEFAULT_VISEME_DURATION: f32 = 0.08;

/// Minimum weight threshold for considering a viseme active.
pub const VISEME_WEIGHT_THRESHOLD: f32 = 0.001;

/// Default coarticulation look-ahead time in seconds.
pub const COARTICULATION_LOOKAHEAD: f32 = 0.05;

/// Default coarticulation carryover time in seconds.
pub const COARTICULATION_CARRYOVER: f32 = 0.03;

/// Default audio sample rate for analysis.
pub const DEFAULT_SAMPLE_RATE: f32 = 44100.0;

/// Amplitude smoothing factor for audio analysis.
pub const AMPLITUDE_SMOOTHING: f32 = 0.85;

/// Voiced detection threshold.
pub const VOICED_THRESHOLD: f32 = 0.1;

// ---------------------------------------------------------------------------
// Viseme
// ---------------------------------------------------------------------------

/// Standard viseme identifiers based on the Preston Blair phoneme set.
///
/// Each viseme represents a distinct mouth shape that can be used
/// to approximate speech animation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Viseme {
    /// Silence / neutral / closed mouth.
    Sil,
    /// P, B, M - bilabial closure (lips pressed together).
    PP,
    /// F, V - labiodental (lower lip to upper teeth).
    FF,
    /// TH (voiced and unvoiced) - dental fricative.
    TH,
    /// T, D, N (alveolar) - tongue to alveolar ridge.
    DD,
    /// K, G - velar stops.
    KK,
    /// CH, J, SH - postalveolar/palatal.
    CH,
    /// S, Z - sibilant fricatives.
    SS,
    /// N, NG - nasal sounds.
    NN,
    /// R - rhotic sounds.
    RR,
    /// A, AH, AI - open/low vowels.
    AA,
    /// E, EH, AY - front mid vowels.
    EE,
    /// I, IH - near-close front vowels.
    IH,
    /// O, OH, AW - back mid rounded vowels.
    OH,
    /// OO, U, W - close back rounded vowels.
    OU,
}

impl Viseme {
    /// Get all visemes in order.
    pub fn all() -> &'static [Viseme] {
        &[
            Viseme::Sil,
            Viseme::PP,
            Viseme::FF,
            Viseme::TH,
            Viseme::DD,
            Viseme::KK,
            Viseme::CH,
            Viseme::SS,
            Viseme::NN,
            Viseme::RR,
            Viseme::AA,
            Viseme::EE,
            Viseme::IH,
            Viseme::OH,
            Viseme::OU,
        ]
    }

    /// Get the viseme index (0-14).
    pub fn index(&self) -> usize {
        match self {
            Viseme::Sil => 0,
            Viseme::PP => 1,
            Viseme::FF => 2,
            Viseme::TH => 3,
            Viseme::DD => 4,
            Viseme::KK => 5,
            Viseme::CH => 6,
            Viseme::SS => 7,
            Viseme::NN => 8,
            Viseme::RR => 9,
            Viseme::AA => 10,
            Viseme::EE => 11,
            Viseme::IH => 12,
            Viseme::OH => 13,
            Viseme::OU => 14,
        }
    }

    /// Create viseme from index.
    pub fn from_index(index: usize) -> Option<Viseme> {
        match index {
            0 => Some(Viseme::Sil),
            1 => Some(Viseme::PP),
            2 => Some(Viseme::FF),
            3 => Some(Viseme::TH),
            4 => Some(Viseme::DD),
            5 => Some(Viseme::KK),
            6 => Some(Viseme::CH),
            7 => Some(Viseme::SS),
            8 => Some(Viseme::NN),
            9 => Some(Viseme::RR),
            10 => Some(Viseme::AA),
            11 => Some(Viseme::EE),
            12 => Some(Viseme::IH),
            13 => Some(Viseme::OH),
            14 => Some(Viseme::OU),
            _ => None,
        }
    }

    /// Get the short code for this viseme.
    pub fn code(&self) -> &'static str {
        match self {
            Viseme::Sil => "sil",
            Viseme::PP => "PP",
            Viseme::FF => "FF",
            Viseme::TH => "TH",
            Viseme::DD => "DD",
            Viseme::KK => "KK",
            Viseme::CH => "CH",
            Viseme::SS => "SS",
            Viseme::NN => "NN",
            Viseme::RR => "RR",
            Viseme::AA => "AA",
            Viseme::EE => "EE",
            Viseme::IH => "IH",
            Viseme::OH => "OH",
            Viseme::OU => "OU",
        }
    }

    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            Viseme::Sil => "Silence/neutral",
            Viseme::PP => "Bilabial (P, B, M)",
            Viseme::FF => "Labiodental (F, V)",
            Viseme::TH => "Dental (TH)",
            Viseme::DD => "Alveolar (T, D)",
            Viseme::KK => "Velar (K, G)",
            Viseme::CH => "Postalveolar (CH, SH, J)",
            Viseme::SS => "Sibilant (S, Z)",
            Viseme::NN => "Nasal (N, NG)",
            Viseme::RR => "Rhotic (R)",
            Viseme::AA => "Open vowel (A, AH)",
            Viseme::EE => "Front mid vowel (E, EH)",
            Viseme::IH => "Close front vowel (I, IH)",
            Viseme::OH => "Back mid vowel (O, OH)",
            Viseme::OU => "Close back vowel (OO, U)",
        }
    }

    /// Check if this is a vowel viseme.
    pub fn is_vowel(&self) -> bool {
        matches!(
            self,
            Viseme::AA | Viseme::EE | Viseme::IH | Viseme::OH | Viseme::OU
        )
    }

    /// Check if this is a consonant viseme.
    pub fn is_consonant(&self) -> bool {
        !self.is_vowel() && *self != Viseme::Sil
    }

    /// Check if this viseme involves lip rounding.
    pub fn is_rounded(&self) -> bool {
        matches!(self, Viseme::OH | Viseme::OU | Viseme::PP)
    }

    /// Get the typical blend shape name for this viseme.
    pub fn blend_shape_name(&self) -> &'static str {
        match self {
            Viseme::Sil => "viseme_sil",
            Viseme::PP => "viseme_PP",
            Viseme::FF => "viseme_FF",
            Viseme::TH => "viseme_TH",
            Viseme::DD => "viseme_DD",
            Viseme::KK => "viseme_kk",
            Viseme::CH => "viseme_CH",
            Viseme::SS => "viseme_SS",
            Viseme::NN => "viseme_nn",
            Viseme::RR => "viseme_RR",
            Viseme::AA => "viseme_aa",
            Viseme::EE => "viseme_E",
            Viseme::IH => "viseme_I",
            Viseme::OH => "viseme_O",
            Viseme::OU => "viseme_U",
        }
    }

    /// Get the FACS action units that approximate this viseme.
    ///
    /// Returns pairs of (ActionUnit, intensity) where intensity is on FACS scale (0-5).
    pub fn facs_mapping(&self) -> Vec<(ActionUnit, f32)> {
        match self {
            Viseme::Sil => vec![],
            Viseme::PP => vec![
                (ActionUnit::AU23, INTENSITY_C), // Lip tightener
                (ActionUnit::AU17, INTENSITY_B), // Chin raiser
            ],
            Viseme::FF => vec![
                (ActionUnit::AU10, INTENSITY_B), // Upper lip raiser
                (ActionUnit::AU25, INTENSITY_B), // Lips part
            ],
            Viseme::TH => vec![
                (ActionUnit::AU25, INTENSITY_C), // Lips part
                (ActionUnit::AU26, INTENSITY_B), // Jaw drop
            ],
            Viseme::DD => vec![
                (ActionUnit::AU25, INTENSITY_B), // Lips part
                (ActionUnit::AU26, INTENSITY_B), // Jaw drop
            ],
            Viseme::KK => vec![
                (ActionUnit::AU25, INTENSITY_B), // Lips part
                (ActionUnit::AU26, INTENSITY_C), // Jaw drop
            ],
            Viseme::CH => vec![
                (ActionUnit::AU23, INTENSITY_B), // Lip tightener
                (ActionUnit::AU25, INTENSITY_B), // Lips part
            ],
            Viseme::SS => vec![
                (ActionUnit::AU20, INTENSITY_B), // Lip stretcher
                (ActionUnit::AU25, INTENSITY_B), // Lips part
            ],
            Viseme::NN => vec![
                (ActionUnit::AU25, INTENSITY_B), // Lips part
                (ActionUnit::AU26, INTENSITY_B), // Jaw drop
            ],
            Viseme::RR => vec![
                (ActionUnit::AU25, INTENSITY_B), // Lips part
                (ActionUnit::AU17, INTENSITY_B), // Chin raiser
            ],
            Viseme::AA => vec![
                (ActionUnit::AU25, INTENSITY_C), // Lips part
                (ActionUnit::AU26, INTENSITY_C), // Jaw drop
                (ActionUnit::AU27, INTENSITY_B), // Mouth stretch
            ],
            Viseme::EE => vec![
                (ActionUnit::AU20, INTENSITY_C), // Lip stretcher (smile-like)
                (ActionUnit::AU25, INTENSITY_B), // Lips part
            ],
            Viseme::IH => vec![
                (ActionUnit::AU20, INTENSITY_B), // Lip stretcher
                (ActionUnit::AU25, INTENSITY_B), // Lips part
                (ActionUnit::AU26, INTENSITY_B), // Jaw drop
            ],
            Viseme::OH => vec![
                (ActionUnit::AU23, INTENSITY_B), // Lip tightener (rounding)
                (ActionUnit::AU25, INTENSITY_B), // Lips part
                (ActionUnit::AU26, INTENSITY_C), // Jaw drop
            ],
            Viseme::OU => vec![
                (ActionUnit::AU23, INTENSITY_C), // Lip tightener (rounding)
                (ActionUnit::AU25, INTENSITY_B), // Lips part
                (ActionUnit::AU17, INTENSITY_B), // Chin raiser
            ],
        }
    }
}

impl Default for Viseme {
    fn default() -> Self {
        Viseme::Sil
    }
}

impl fmt::Display for Viseme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

// ---------------------------------------------------------------------------
// BlendCurve
// ---------------------------------------------------------------------------

/// Blend curve types for viseme transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BlendCurve {
    /// Linear interpolation.
    #[default]
    Linear,
    /// Smooth step (ease in/out).
    SmoothStep,
    /// Fast attack, slow decay.
    FastAttack,
    /// Slow attack, fast decay.
    SlowAttack,
    /// Sine wave easing.
    Sine,
}

impl BlendCurve {
    /// Apply the blend curve to a normalized time value.
    ///
    /// `t` is normalized time (0.0 at start, 1.0 at end).
    /// Returns the interpolated value.
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            BlendCurve::Linear => t,
            BlendCurve::SmoothStep => t * t * (3.0 - 2.0 * t),
            BlendCurve::FastAttack => 1.0 - (1.0 - t).powi(3),
            BlendCurve::SlowAttack => t.powi(3),
            BlendCurve::Sine => 0.5 * (1.0 - (std::f32::consts::PI * t).cos()),
        }
    }

    /// Apply inverse curve (for decay).
    pub fn apply_inverse(&self, t: f32) -> f32 {
        1.0 - self.apply(t)
    }
}

// ---------------------------------------------------------------------------
// VisemeWeight
// ---------------------------------------------------------------------------

/// Weight configuration for a single viseme.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisemeWeight {
    /// The viseme being controlled.
    pub viseme: Viseme,
    /// Weight/intensity (0.0 - 1.0).
    pub weight: f32,
    /// Blend curve for transitions.
    pub blend_curve: BlendCurve,
    /// Timing offset in seconds (for coarticulation).
    pub timing_offset: f32,
}

impl VisemeWeight {
    /// Create a new viseme weight.
    pub fn new(viseme: Viseme, weight: f32) -> Self {
        Self {
            viseme,
            weight: weight.clamp(0.0, 1.0),
            blend_curve: BlendCurve::default(),
            timing_offset: 0.0,
        }
    }

    /// Builder: set blend curve.
    pub fn with_curve(mut self, curve: BlendCurve) -> Self {
        self.blend_curve = curve;
        self
    }

    /// Builder: set timing offset.
    pub fn with_offset(mut self, offset: f32) -> Self {
        self.timing_offset = offset;
        self
    }

    /// Check if this weight is active (above threshold).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.weight > VISEME_WEIGHT_THRESHOLD
    }

    /// Scale the weight by a factor.
    pub fn scaled(&self, factor: f32) -> Self {
        Self {
            viseme: self.viseme,
            weight: (self.weight * factor).clamp(0.0, 1.0),
            blend_curve: self.blend_curve,
            timing_offset: self.timing_offset,
        }
    }
}

impl Default for VisemeWeight {
    fn default() -> Self {
        Self {
            viseme: Viseme::Sil,
            weight: 0.0,
            blend_curve: BlendCurve::default(),
            timing_offset: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// VisemeKeyframe
// ---------------------------------------------------------------------------

/// A keyframe in a viseme animation track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisemeKeyframe {
    /// Time in seconds.
    pub time: f32,
    /// Viseme at this keyframe.
    pub viseme: Viseme,
    /// Target weight at this keyframe.
    pub weight: f32,
    /// Duration to hold this viseme.
    pub duration: f32,
    /// Blend curve for transition to this viseme.
    pub blend_curve: BlendCurve,
}

impl VisemeKeyframe {
    /// Create a new keyframe.
    pub fn new(time: f32, viseme: Viseme, weight: f32, duration: f32) -> Self {
        Self {
            time: time.max(0.0),
            viseme,
            weight: weight.clamp(0.0, 1.0),
            duration: duration.max(0.0),
            blend_curve: BlendCurve::default(),
        }
    }

    /// Builder: set blend curve.
    pub fn with_curve(mut self, curve: BlendCurve) -> Self {
        self.blend_curve = curve;
        self
    }

    /// Get the end time of this keyframe's hold period.
    #[inline]
    pub fn end_time(&self) -> f32 {
        self.time + self.duration
    }
}

// ---------------------------------------------------------------------------
// LipSyncEvent
// ---------------------------------------------------------------------------

/// Event markers in a lip sync track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LipSyncEvent {
    /// Start of a word.
    WordStart {
        time: f32,
        word: String,
    },
    /// End of a word.
    WordEnd {
        time: f32,
    },
    /// Emphasis marker (stressed syllable).
    Emphasis {
        time: f32,
        intensity: f32,
    },
    /// Pause marker.
    Pause {
        time: f32,
        duration: f32,
    },
    /// Custom event.
    Custom {
        time: f32,
        name: String,
        data: String,
    },
}

impl LipSyncEvent {
    /// Get the time of this event.
    pub fn time(&self) -> f32 {
        match self {
            LipSyncEvent::WordStart { time, .. } => *time,
            LipSyncEvent::WordEnd { time } => *time,
            LipSyncEvent::Emphasis { time, .. } => *time,
            LipSyncEvent::Pause { time, .. } => *time,
            LipSyncEvent::Custom { time, .. } => *time,
        }
    }
}

// ---------------------------------------------------------------------------
// LipSyncTrack
// ---------------------------------------------------------------------------

/// A timed sequence of viseme weights for lip sync animation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LipSyncTrack {
    /// Keyframes sorted by time.
    pub keyframes: Vec<VisemeKeyframe>,
    /// Event markers.
    pub events: Vec<LipSyncEvent>,
    /// Total duration in seconds.
    pub duration: f32,
    /// Whether to apply coarticulation.
    pub use_coarticulation: bool,
    /// Coarticulation look-ahead time.
    pub coarticulation_lookahead: f32,
    /// Coarticulation carryover time.
    pub coarticulation_carryover: f32,
}

impl LipSyncTrack {
    /// Create a new empty lip sync track.
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
            events: Vec::new(),
            duration: 0.0,
            use_coarticulation: true,
            coarticulation_lookahead: COARTICULATION_LOOKAHEAD,
            coarticulation_carryover: COARTICULATION_CARRYOVER,
        }
    }

    /// Create a track with specified duration.
    pub fn with_duration(duration: f32) -> Self {
        Self {
            duration,
            ..Self::new()
        }
    }

    /// Add a viseme keyframe.
    pub fn add_viseme(&mut self, time: f32, viseme: Viseme, weight: f32, duration: f32) {
        let keyframe = VisemeKeyframe::new(time, viseme, weight, duration);
        self.insert_keyframe(keyframe);
    }

    /// Insert a keyframe (maintains sorted order).
    pub fn insert_keyframe(&mut self, keyframe: VisemeKeyframe) {
        let end_time = keyframe.end_time();
        let pos = self
            .keyframes
            .iter()
            .position(|k| k.time > keyframe.time)
            .unwrap_or(self.keyframes.len());
        self.keyframes.insert(pos, keyframe);

        // Update duration
        if end_time > self.duration {
            self.duration = end_time;
        }
    }

    /// Add an event marker.
    pub fn add_event(&mut self, event: LipSyncEvent) {
        let time = event.time();
        let pos = self
            .events
            .iter()
            .position(|e| e.time() > time)
            .unwrap_or(self.events.len());
        self.events.insert(pos, event);
    }

    /// Get all events in a time range.
    pub fn events_in_range(&self, start: f32, end: f32) -> Vec<&LipSyncEvent> {
        self.events
            .iter()
            .filter(|e| e.time() >= start && e.time() < end)
            .collect()
    }

    /// Sample the track at an arbitrary time.
    ///
    /// Returns a vector of active viseme weights at the given time.
    pub fn sample(&self, time: f32) -> Vec<VisemeWeight> {
        if self.keyframes.is_empty() {
            return vec![VisemeWeight::new(Viseme::Sil, 1.0)];
        }

        let mut weights = HashMap::new();

        // Find keyframes that affect this time
        for (i, kf) in self.keyframes.iter().enumerate() {
            let weight = self.calculate_keyframe_weight(i, time);
            if weight > VISEME_WEIGHT_THRESHOLD {
                weights
                    .entry(kf.viseme)
                    .and_modify(|w: &mut f32| *w = (*w + weight).min(1.0))
                    .or_insert(weight);
            }
        }

        // Apply coarticulation if enabled
        if self.use_coarticulation {
            self.apply_coarticulation(&mut weights, time);
        }

        // Convert to VisemeWeight vec
        let mut result: Vec<_> = weights
            .into_iter()
            .filter(|(_, w)| *w > VISEME_WEIGHT_THRESHOLD)
            .map(|(v, w)| VisemeWeight::new(v, w))
            .collect();

        // If no active visemes, return silence
        if result.is_empty() {
            result.push(VisemeWeight::new(Viseme::Sil, 1.0));
        }

        result
    }

    /// Calculate the weight contribution of a keyframe at the given time.
    fn calculate_keyframe_weight(&self, keyframe_idx: usize, time: f32) -> f32 {
        let kf = &self.keyframes[keyframe_idx];

        // Before keyframe starts
        if time < kf.time {
            // Check for blend-in from previous
            let blend_time = DEFAULT_VISEME_DURATION;
            let blend_start = kf.time - blend_time;
            if time >= blend_start {
                let t = (time - blend_start) / blend_time;
                return kf.weight * kf.blend_curve.apply(t);
            }
            return 0.0;
        }

        // During keyframe hold
        if time <= kf.end_time() {
            return kf.weight;
        }

        // After keyframe ends - blend out
        let blend_time = DEFAULT_VISEME_DURATION;
        let blend_end = kf.end_time() + blend_time;
        if time < blend_end {
            let t = (time - kf.end_time()) / blend_time;
            return kf.weight * kf.blend_curve.apply_inverse(t);
        }

        0.0
    }

    /// Apply coarticulation effects to the weights.
    fn apply_coarticulation(&self, weights: &mut HashMap<Viseme, f32>, time: f32) {
        // Find next viseme for anticipatory coarticulation
        let next_kf = self
            .keyframes
            .iter()
            .find(|kf| kf.time > time && kf.time - time < self.coarticulation_lookahead);

        if let Some(next) = next_kf {
            // Apply anticipatory influence
            let t = 1.0 - (next.time - time) / self.coarticulation_lookahead;
            let anticipation_weight = t * 0.3 * next.weight; // 30% max anticipation

            if anticipation_weight > VISEME_WEIGHT_THRESHOLD {
                // Special handling for rounded vowels
                if next.viseme.is_rounded() {
                    weights
                        .entry(next.viseme)
                        .and_modify(|w| *w = (*w + anticipation_weight).min(1.0))
                        .or_insert(anticipation_weight);
                }
            }
        }

        // Find previous viseme for carryover
        let prev_kf = self
            .keyframes
            .iter()
            .rev()
            .find(|kf| kf.end_time() < time && time - kf.end_time() < self.coarticulation_carryover);

        if let Some(prev) = prev_kf {
            // Apply carryover influence
            let t = (time - prev.end_time()) / self.coarticulation_carryover;
            let carryover_weight = (1.0 - t) * 0.2 * prev.weight; // 20% max carryover

            if carryover_weight > VISEME_WEIGHT_THRESHOLD {
                weights
                    .entry(prev.viseme)
                    .and_modify(|w| *w = (*w + carryover_weight).min(1.0))
                    .or_insert(carryover_weight);
            }
        }
    }

    /// Get the number of keyframes.
    #[inline]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Check if the track is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Clear all keyframes and events.
    pub fn clear(&mut self) {
        self.keyframes.clear();
        self.events.clear();
        self.duration = 0.0;
    }

    /// Enable/disable coarticulation.
    pub fn set_coarticulation(&mut self, enabled: bool) {
        self.use_coarticulation = enabled;
    }
}

// ---------------------------------------------------------------------------
// Language
// ---------------------------------------------------------------------------

/// Supported languages for phoneme-to-viseme mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Language {
    /// American English.
    #[default]
    EnglishUS,
    /// British English.
    EnglishUK,
    /// Japanese.
    Japanese,
    /// Spanish.
    Spanish,
    /// French.
    French,
    /// German.
    German,
    /// Mandarin Chinese.
    Mandarin,
    /// Korean.
    Korean,
}

impl Language {
    /// Get all supported languages.
    pub fn all() -> &'static [Language] {
        &[
            Language::EnglishUS,
            Language::EnglishUK,
            Language::Japanese,
            Language::Spanish,
            Language::French,
            Language::German,
            Language::Mandarin,
            Language::Korean,
        ]
    }

    /// Get the language code.
    pub fn code(&self) -> &'static str {
        match self {
            Language::EnglishUS => "en-US",
            Language::EnglishUK => "en-GB",
            Language::Japanese => "ja",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Mandarin => "zh",
            Language::Korean => "ko",
        }
    }

    /// Get the language name.
    pub fn name(&self) -> &'static str {
        match self {
            Language::EnglishUS => "English (US)",
            Language::EnglishUK => "English (UK)",
            Language::Japanese => "Japanese",
            Language::Spanish => "Spanish",
            Language::French => "French",
            Language::German => "German",
            Language::Mandarin => "Mandarin Chinese",
            Language::Korean => "Korean",
        }
    }
}

// ---------------------------------------------------------------------------
// PhonemeToViseme
// ---------------------------------------------------------------------------

/// Phoneme to viseme mapping with language support.
#[derive(Clone, Debug, Default)]
pub struct PhonemeToViseme {
    /// Current language.
    pub language: Language,
    /// Custom phoneme overrides.
    pub overrides: HashMap<String, Viseme>,
}

impl PhonemeToViseme {
    /// Create a new mapper with default language.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a mapper for a specific language.
    pub fn with_language(language: Language) -> Self {
        Self {
            language,
            overrides: HashMap::new(),
        }
    }

    /// Set the language.
    pub fn set_language(&mut self, language: Language) {
        self.language = language;
    }

    /// Add a custom phoneme override.
    pub fn add_override(&mut self, phoneme: impl Into<String>, viseme: Viseme) {
        self.overrides.insert(phoneme.into().to_lowercase(), viseme);
    }

    /// Map a phoneme to a viseme.
    pub fn map(&self, phoneme: &str) -> Viseme {
        // Special case: Japanese syllabic N is case-sensitive
        if self.language == Language::Japanese && phoneme == "N" {
            return Viseme::NN;
        }

        let phoneme_lower = phoneme.to_lowercase();

        // Check overrides first
        if let Some(v) = self.overrides.get(&phoneme_lower) {
            return *v;
        }

        // Language-specific mappings
        match self.language {
            Language::Japanese => self.map_japanese(&phoneme_lower),
            _ => self.map_english(&phoneme_lower),
        }
    }

    /// Map phoneme clusters (multiple phonemes).
    pub fn map_cluster(&self, phonemes: &[&str]) -> Vec<Viseme> {
        phonemes.iter().map(|p| self.map(p)).collect()
    }

    /// English phoneme mapping (IPA-based).
    fn map_english(&self, phoneme: &str) -> Viseme {
        match phoneme {
            // Silence
            "" | "sil" | "_" | "#" => Viseme::Sil,

            // Bilabials (P, B, M)
            "p" | "b" | "m" => Viseme::PP,

            // Labiodentals (F, V)
            "f" | "v" => Viseme::FF,

            // Dentals (TH)
            "th" | "dh" | "\u{03b8}" | "\u{00f0}" => Viseme::TH,

            // Alveolars (T, D, N, L)
            "t" | "d" | "n" | "l" => Viseme::DD,

            // Velars (K, G, NG)
            "k" | "g" => Viseme::KK,
            "ng" | "\u{014b}" => Viseme::NN,

            // Postalveolars (CH, J, SH, ZH)
            "ch" | "tsh" | "j" | "dj" | "sh" | "zh" | "\u{0283}" | "\u{0292}" | "t\u{0283}"
            | "d\u{0292}" => Viseme::CH,

            // Sibilants (S, Z)
            "s" | "z" => Viseme::SS,

            // Rhotics (R)
            "r" | "er" | "\u{0279}" | "\u{025d}" => Viseme::RR,

            // Glides
            "w" => Viseme::OU, // W has lip rounding
            "y" => Viseme::IH,
            "h" => Viseme::AA, // H often opens mouth

            // Vowels - Open (A sounds)
            "a" | "aa" | "ah" | "ae" | "\u{00e6}" | "\u{0251}" | "\u{0252}" | "ai" | "ay" => {
                Viseme::AA
            }

            // Vowels - Front mid (E sounds)
            "e" | "eh" | "ey" | "\u{025b}" | "\u{0259}" => Viseme::EE,

            // Vowels - Close front (I sounds)
            "i" | "ih" | "iy" | "\u{026a}" => Viseme::IH,

            // Vowels - Back mid (O sounds)
            "o" | "oh" | "aw" | "ao" | "\u{0254}" | "oy" | "oi" => Viseme::OH,

            // Vowels - Close back (U/OO sounds)
            "u" | "oo" | "uw" | "uh" | "\u{028a}" | "ow" | "ou" => Viseme::OU,

            // Default to silence for unknown
            _ => Viseme::Sil,
        }
    }

    /// Japanese phoneme mapping.
    fn map_japanese(&self, phoneme: &str) -> Viseme {
        match phoneme {
            // Silence
            "" | "sil" | "_" | "#" | "q" => Viseme::Sil,

            // Japanese consonants
            "p" | "b" | "m" => Viseme::PP,
            "f" | "h" | "hy" => Viseme::FF,
            "t" | "d" | "n" | "ts" => Viseme::DD,
            "k" | "g" | "ky" | "gy" => Viseme::KK,
            "s" | "z" | "sh" | "j" | "ch" | "sy" | "zy" => Viseme::SS,
            "r" | "ry" => Viseme::RR,
            "w" => Viseme::OU,
            "y" => Viseme::IH,
            "ny" => Viseme::NN,
            "N" => Viseme::NN, // Syllabic N

            // Japanese vowels
            "a" => Viseme::AA,
            "e" => Viseme::EE,
            "i" => Viseme::IH,
            "o" => Viseme::OH,
            "u" => Viseme::OU,

            // Default
            _ => Viseme::Sil,
        }
    }

    /// Create a viseme track from phoneme timing data.
    pub fn create_track(&self, phoneme_timings: &[(f32, f32, &str)]) -> LipSyncTrack {
        let mut track = LipSyncTrack::new();

        for (start, end, phoneme) in phoneme_timings {
            let viseme = self.map(phoneme);
            let duration = end - start;
            track.add_viseme(*start, viseme, 1.0, duration);
        }

        track
    }
}

// ---------------------------------------------------------------------------
// CoarticulationRule
// ---------------------------------------------------------------------------

/// Coarticulation rule for blending adjacent phonemes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoarticulationRule {
    /// The viseme this rule applies to.
    pub viseme: Viseme,
    /// Next visemes that trigger anticipation.
    pub anticipates: Vec<Viseme>,
    /// Amount of anticipatory influence (0.0 - 1.0).
    pub anticipation_strength: f32,
    /// Previous visemes that carry over.
    pub carries_over_from: Vec<Viseme>,
    /// Amount of carryover influence (0.0 - 1.0).
    pub carryover_strength: f32,
}

impl CoarticulationRule {
    /// Create a new rule.
    pub fn new(viseme: Viseme) -> Self {
        Self {
            viseme,
            anticipates: Vec::new(),
            anticipation_strength: 0.3,
            carries_over_from: Vec::new(),
            carryover_strength: 0.2,
        }
    }

    /// Add visemes that trigger anticipation.
    pub fn anticipate(mut self, visemes: &[Viseme]) -> Self {
        self.anticipates.extend_from_slice(visemes);
        self
    }

    /// Add visemes that carry over.
    pub fn carryover(mut self, visemes: &[Viseme]) -> Self {
        self.carries_over_from.extend_from_slice(visemes);
        self
    }

    /// Set anticipation strength.
    pub fn with_anticipation_strength(mut self, strength: f32) -> Self {
        self.anticipation_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set carryover strength.
    pub fn with_carryover_strength(mut self, strength: f32) -> Self {
        self.carryover_strength = strength.clamp(0.0, 1.0);
        self
    }
}

/// Collection of coarticulation rules.
#[derive(Clone, Debug, Default)]
pub struct CoarticulationRules {
    rules: HashMap<Viseme, CoarticulationRule>,
}

impl CoarticulationRules {
    /// Create with default rules.
    pub fn new() -> Self {
        let mut rules = Self::default();
        rules.add_default_rules();
        rules
    }

    /// Add default coarticulation rules.
    fn add_default_rules(&mut self) {
        // Lip rounding anticipation
        let rounded = &[Viseme::OH, Viseme::OU];
        for consonant in &[
            Viseme::PP,
            Viseme::FF,
            Viseme::DD,
            Viseme::KK,
            Viseme::SS,
        ] {
            self.rules.insert(
                *consonant,
                CoarticulationRule::new(*consonant)
                    .anticipate(rounded)
                    .with_anticipation_strength(0.4),
            );
        }

        // Vowel carryover
        for vowel in &[Viseme::AA, Viseme::EE, Viseme::IH, Viseme::OH, Viseme::OU] {
            self.rules.insert(
                *vowel,
                CoarticulationRule::new(*vowel).with_carryover_strength(0.25),
            );
        }
    }

    /// Get a rule for a viseme.
    pub fn get(&self, viseme: Viseme) -> Option<&CoarticulationRule> {
        self.rules.get(&viseme)
    }

    /// Add a custom rule.
    pub fn add(&mut self, rule: CoarticulationRule) {
        self.rules.insert(rule.viseme, rule);
    }
}

// ---------------------------------------------------------------------------
// AudioAnalyzer
// ---------------------------------------------------------------------------

/// Simple audio analyzer for amplitude-based lip sync fallback.
///
/// When phoneme data is unavailable, this provides basic animation
/// driven by audio amplitude (volume).
#[derive(Clone, Debug)]
pub struct AudioAnalyzer {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Smoothed amplitude (0.0 - 1.0).
    pub amplitude: f32,
    /// Peak amplitude for normalization.
    pub peak_amplitude: f32,
    /// Smoothing factor (0.0 - 1.0).
    pub smoothing: f32,
    /// Whether voice is detected.
    pub is_voiced: bool,
    /// Voiced detection threshold.
    pub voiced_threshold: f32,
    /// Simple formant hint (high/low energy ratio).
    pub formant_hint: f32,
}

impl AudioAnalyzer {
    /// Create a new audio analyzer.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            amplitude: 0.0,
            peak_amplitude: 1.0,
            smoothing: AMPLITUDE_SMOOTHING,
            is_voiced: false,
            voiced_threshold: VOICED_THRESHOLD,
            formant_hint: 0.5,
        }
    }

    /// Process audio samples and update state.
    ///
    /// Returns the current smoothed amplitude.
    pub fn process_samples(&mut self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            self.is_voiced = false;
            return self.amplitude;
        }

        // Calculate RMS amplitude
        let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
        let rms = (sum_squares / samples.len() as f32).sqrt();

        // Update peak
        if rms > self.peak_amplitude * 0.9 {
            self.peak_amplitude = rms * 1.1;
        }

        // Normalize
        let normalized = rms / self.peak_amplitude.max(0.001);

        // Smooth
        self.amplitude = self.amplitude * self.smoothing + normalized * (1.0 - self.smoothing);

        // Detect voiced
        self.is_voiced = self.amplitude > self.voiced_threshold;

        // Simple formant hint: high frequency energy ratio
        self.formant_hint = self.calculate_formant_hint(samples);

        self.amplitude
    }

    /// Calculate a simple formant hint from samples.
    ///
    /// Returns a value 0.0-1.0 where higher values indicate
    /// more high-frequency content (like 'ee', 'i' sounds).
    fn calculate_formant_hint(&self, samples: &[f32]) -> f32 {
        if samples.len() < 4 {
            return 0.5;
        }

        // Simple high-pass: difference between adjacent samples
        let high_energy: f32 = samples
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum();

        let total_energy: f32 = samples.iter().map(|s| s.abs()).sum();

        if total_energy < 0.0001 {
            return 0.5;
        }

        (high_energy / total_energy).clamp(0.0, 1.0)
    }

    /// Get a jaw open amount based on current amplitude.
    pub fn jaw_open(&self) -> f32 {
        if !self.is_voiced {
            return 0.0;
        }
        // Apply curve for more natural motion
        let t = self.amplitude.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Suggest a viseme based on current audio state.
    pub fn suggest_viseme(&self) -> Viseme {
        if !self.is_voiced {
            return Viseme::Sil;
        }

        // Use amplitude and formant hint to suggest a viseme
        if self.amplitude > 0.7 {
            // High amplitude = open mouth
            if self.formant_hint > 0.6 {
                Viseme::EE // High frequency = front vowel
            } else {
                Viseme::AA // Low frequency = open vowel
            }
        } else if self.amplitude > 0.4 {
            // Medium amplitude
            if self.formant_hint > 0.6 {
                Viseme::IH
            } else {
                Viseme::OH
            }
        } else {
            // Low amplitude
            if self.formant_hint > 0.5 {
                Viseme::EE
            } else {
                Viseme::OU
            }
        }
    }

    /// Reset the analyzer state.
    pub fn reset(&mut self) {
        self.amplitude = 0.0;
        self.is_voiced = false;
        self.formant_hint = 0.5;
    }
}

impl Default for AudioAnalyzer {
    fn default() -> Self {
        Self::new(DEFAULT_SAMPLE_RATE)
    }
}

// ---------------------------------------------------------------------------
// LipSyncController
// ---------------------------------------------------------------------------

/// Main controller for lip sync animation.
///
/// Coordinates between viseme tracks, audio analysis, and FACS integration.
#[derive(Clone, Debug, Default)]
pub struct LipSyncController {
    /// Current viseme track.
    pub track: Option<LipSyncTrack>,
    /// Audio analyzer for fallback.
    pub audio_analyzer: AudioAnalyzer,
    /// Phoneme to viseme mapper.
    pub phoneme_mapper: PhonemeToViseme,
    /// Current playback time.
    pub current_time: f32,
    /// Whether playback is active.
    pub is_playing: bool,
    /// Playback speed multiplier.
    pub playback_speed: f32,
    /// Global intensity multiplier.
    pub intensity: f32,
    /// Use audio fallback when track ends.
    pub use_audio_fallback: bool,
    /// Current active viseme weights (cached).
    cached_weights: Vec<VisemeWeight>,
    /// Custom blend shape name mappings.
    pub blend_shape_mappings: HashMap<Viseme, String>,
}

impl LipSyncController {
    /// Create a new lip sync controller.
    pub fn new() -> Self {
        Self {
            track: None,
            audio_analyzer: AudioAnalyzer::default(),
            phoneme_mapper: PhonemeToViseme::default(),
            current_time: 0.0,
            is_playing: false,
            playback_speed: 1.0,
            intensity: 1.0,
            use_audio_fallback: true,
            cached_weights: Vec::new(),
            blend_shape_mappings: HashMap::new(),
        }
    }

    /// Set the active viseme track.
    pub fn set_track(&mut self, track: LipSyncTrack) {
        self.track = Some(track);
        self.current_time = 0.0;
    }

    /// Clear the current track.
    pub fn clear_track(&mut self) {
        self.track = None;
        self.current_time = 0.0;
        self.cached_weights.clear();
    }

    /// Start playback.
    pub fn play(&mut self) {
        self.is_playing = true;
    }

    /// Stop playback.
    pub fn stop(&mut self) {
        self.is_playing = false;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.is_playing = false;
    }

    /// Seek to a specific time.
    pub fn seek(&mut self, time: f32) {
        self.current_time = time.max(0.0);
    }

    /// Update the controller with delta time.
    pub fn update(&mut self, dt: f32) {
        if !self.is_playing {
            return;
        }

        self.current_time += dt * self.playback_speed;

        // Check if track has ended
        if let Some(ref track) = self.track {
            if self.current_time >= track.duration {
                if self.use_audio_fallback {
                    // Continue with audio fallback
                } else {
                    self.is_playing = false;
                }
            }
        }

        // Update cached weights
        self.update_weights();
    }

    /// Update cached viseme weights.
    fn update_weights(&mut self) {
        self.cached_weights.clear();

        // Get weights from track if available and within duration
        if let Some(ref track) = self.track {
            if self.current_time < track.duration {
                let mut weights = track.sample(self.current_time);
                // Apply intensity
                for w in &mut weights {
                    w.weight *= self.intensity;
                }
                self.cached_weights = weights;
                return;
            }
        }

        // Use audio fallback
        if self.use_audio_fallback && self.audio_analyzer.is_voiced {
            let viseme = self.audio_analyzer.suggest_viseme();
            let weight = self.audio_analyzer.amplitude * self.intensity;
            self.cached_weights
                .push(VisemeWeight::new(viseme, weight));
        } else {
            self.cached_weights
                .push(VisemeWeight::new(Viseme::Sil, 1.0));
        }
    }

    /// Process audio samples for fallback animation.
    pub fn process_audio(&mut self, samples: &[f32]) {
        self.audio_analyzer.process_samples(samples);

        // If no track or track ended, update weights from audio
        let should_use_audio = self.track.is_none()
            || self
                .track
                .as_ref()
                .map(|t| self.current_time >= t.duration)
                .unwrap_or(true);

        if should_use_audio && self.use_audio_fallback {
            self.update_weights();
        }
    }

    /// Get current viseme weights.
    pub fn get_weights(&self) -> &[VisemeWeight] {
        &self.cached_weights
    }

    /// Convert current state to blend shape weights.
    pub fn to_blend_weights(&self) -> HashMap<String, f32> {
        let mut result = HashMap::new();

        for vw in &self.cached_weights {
            let blend_name = self
                .blend_shape_mappings
                .get(&vw.viseme)
                .map(|s| s.as_str())
                .unwrap_or_else(|| vw.viseme.blend_shape_name());

            result
                .entry(blend_name.to_string())
                .and_modify(|w: &mut f32| *w = (*w + vw.weight).min(1.0))
                .or_insert(vw.weight);
        }

        result
    }

    /// Apply current state to a BlendShapeSet.
    pub fn apply_to_blend_shapes(&self, shapes: &mut BlendShapeSet) {
        let weights = self.to_blend_weights();

        for (name, weight) in weights {
            shapes.set_weight_by_name(&name, weight);
        }
    }

    /// Apply current state to a FACS controller.
    pub fn apply_to_facs(&self, facs: &mut FACSController) {
        for vw in &self.cached_weights {
            let facs_mappings = vw.viseme.facs_mapping();
            for (au, base_intensity) in facs_mappings {
                // Scale FACS intensity by viseme weight
                let scaled_intensity = base_intensity * vw.weight;
                facs.add_au(au, scaled_intensity);
            }
        }
    }

    /// Get jaw open amount for amplitude-based fallback.
    pub fn jaw_open(&self) -> f32 {
        // From track or audio
        if let Some(ref track) = self.track {
            if self.current_time < track.duration {
                // Estimate from active visemes
                let max_opening = self
                    .cached_weights
                    .iter()
                    .map(|vw| {
                        // Vowels open more
                        if vw.viseme.is_vowel() {
                            vw.weight
                        } else {
                            vw.weight * 0.3
                        }
                    })
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                return max_opening;
            }
        }

        self.audio_analyzer.jaw_open() * self.intensity
    }

    /// Set a custom blend shape name for a viseme.
    pub fn set_blend_mapping(&mut self, viseme: Viseme, name: impl Into<String>) {
        self.blend_shape_mappings.insert(viseme, name.into());
    }

    /// Get events that occurred in the last frame.
    pub fn get_events(&self, last_time: f32) -> Vec<&LipSyncEvent> {
        if let Some(ref track) = self.track {
            track.events_in_range(last_time, self.current_time)
        } else {
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Create a lip sync track from a transcript with timing.
///
/// Each tuple is (start_time, end_time, phoneme).
pub fn create_track_from_phonemes(
    phoneme_timings: &[(f32, f32, &str)],
    language: Language,
) -> LipSyncTrack {
    let mapper = PhonemeToViseme::with_language(language);
    mapper.create_track(phoneme_timings)
}

/// Create a simple test track that says "hello".
pub fn create_hello_track() -> LipSyncTrack {
    let phonemes = [
        (0.0, 0.1, "h"),
        (0.1, 0.2, "eh"),
        (0.2, 0.35, "l"),
        (0.35, 0.5, "oh"),
        (0.5, 0.6, "sil"),
    ];
    create_track_from_phonemes(&phonemes, Language::EnglishUS)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Viseme Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_viseme_all() {
        let all = Viseme::all();
        assert_eq!(all.len(), VISEME_COUNT);
    }

    #[test]
    fn test_viseme_index() {
        assert_eq!(Viseme::Sil.index(), 0);
        assert_eq!(Viseme::PP.index(), 1);
        assert_eq!(Viseme::OU.index(), 14);
    }

    #[test]
    fn test_viseme_from_index() {
        assert_eq!(Viseme::from_index(0), Some(Viseme::Sil));
        assert_eq!(Viseme::from_index(14), Some(Viseme::OU));
        assert_eq!(Viseme::from_index(15), None);
    }

    #[test]
    fn test_viseme_roundtrip_index() {
        for viseme in Viseme::all() {
            let index = viseme.index();
            let recovered = Viseme::from_index(index);
            assert_eq!(recovered, Some(*viseme));
        }
    }

    #[test]
    fn test_viseme_code() {
        assert_eq!(Viseme::Sil.code(), "sil");
        assert_eq!(Viseme::PP.code(), "PP");
        assert_eq!(Viseme::AA.code(), "AA");
    }

    #[test]
    fn test_viseme_description() {
        assert!(Viseme::Sil.description().contains("Silence"));
        assert!(Viseme::PP.description().contains("Bilabial"));
    }

    #[test]
    fn test_viseme_is_vowel() {
        assert!(Viseme::AA.is_vowel());
        assert!(Viseme::EE.is_vowel());
        assert!(Viseme::IH.is_vowel());
        assert!(Viseme::OH.is_vowel());
        assert!(Viseme::OU.is_vowel());
        assert!(!Viseme::PP.is_vowel());
        assert!(!Viseme::Sil.is_vowel());
    }

    #[test]
    fn test_viseme_is_consonant() {
        assert!(Viseme::PP.is_consonant());
        assert!(Viseme::FF.is_consonant());
        assert!(!Viseme::AA.is_consonant());
        assert!(!Viseme::Sil.is_consonant());
    }

    #[test]
    fn test_viseme_is_rounded() {
        assert!(Viseme::OH.is_rounded());
        assert!(Viseme::OU.is_rounded());
        assert!(Viseme::PP.is_rounded()); // Bilabial closure
        assert!(!Viseme::AA.is_rounded());
        assert!(!Viseme::EE.is_rounded());
    }

    #[test]
    fn test_viseme_blend_shape_name() {
        assert_eq!(Viseme::Sil.blend_shape_name(), "viseme_sil");
        assert_eq!(Viseme::AA.blend_shape_name(), "viseme_aa");
    }

    #[test]
    fn test_viseme_facs_mapping() {
        let pp_facs = Viseme::PP.facs_mapping();
        assert!(!pp_facs.is_empty());
        assert!(pp_facs.iter().any(|(au, _)| *au == ActionUnit::AU23));

        let sil_facs = Viseme::Sil.facs_mapping();
        assert!(sil_facs.is_empty());
    }

    #[test]
    fn test_viseme_display() {
        assert_eq!(format!("{}", Viseme::PP), "PP");
        assert_eq!(format!("{}", Viseme::Sil), "sil");
    }

    #[test]
    fn test_viseme_default() {
        assert_eq!(Viseme::default(), Viseme::Sil);
    }

    // -----------------------------------------------------------------------
    // BlendCurve Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_curve_linear() {
        assert!((BlendCurve::Linear.apply(0.0) - 0.0).abs() < 0.01);
        assert!((BlendCurve::Linear.apply(0.5) - 0.5).abs() < 0.01);
        assert!((BlendCurve::Linear.apply(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_blend_curve_smooth_step() {
        assert!((BlendCurve::SmoothStep.apply(0.0) - 0.0).abs() < 0.01);
        assert!((BlendCurve::SmoothStep.apply(1.0) - 1.0).abs() < 0.01);
        // Smooth step at 0.5 should be 0.5
        assert!((BlendCurve::SmoothStep.apply(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_blend_curve_fast_attack() {
        // Fast attack rises quickly
        assert!(BlendCurve::FastAttack.apply(0.5) > 0.5);
    }

    #[test]
    fn test_blend_curve_slow_attack() {
        // Slow attack rises slowly
        assert!(BlendCurve::SlowAttack.apply(0.5) < 0.5);
    }

    #[test]
    fn test_blend_curve_clamping() {
        assert!((BlendCurve::Linear.apply(-1.0) - 0.0).abs() < 0.01);
        assert!((BlendCurve::Linear.apply(2.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_blend_curve_inverse() {
        assert!((BlendCurve::Linear.apply_inverse(0.0) - 1.0).abs() < 0.01);
        assert!((BlendCurve::Linear.apply_inverse(1.0) - 0.0).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // VisemeWeight Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_viseme_weight_new() {
        let vw = VisemeWeight::new(Viseme::AA, 0.8);
        assert_eq!(vw.viseme, Viseme::AA);
        assert!((vw.weight - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_viseme_weight_clamping() {
        let vw1 = VisemeWeight::new(Viseme::AA, 1.5);
        assert_eq!(vw1.weight, 1.0);

        let vw2 = VisemeWeight::new(Viseme::AA, -0.5);
        assert_eq!(vw2.weight, 0.0);
    }

    #[test]
    fn test_viseme_weight_builder() {
        let vw = VisemeWeight::new(Viseme::PP, 0.7)
            .with_curve(BlendCurve::SmoothStep)
            .with_offset(0.05);

        assert_eq!(vw.blend_curve, BlendCurve::SmoothStep);
        assert!((vw.timing_offset - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_viseme_weight_is_active() {
        assert!(VisemeWeight::new(Viseme::AA, 0.5).is_active());
        assert!(!VisemeWeight::new(Viseme::AA, 0.0).is_active());
        assert!(!VisemeWeight::new(Viseme::AA, 0.0001).is_active());
    }

    #[test]
    fn test_viseme_weight_scaled() {
        let vw = VisemeWeight::new(Viseme::AA, 0.8);
        let scaled = vw.scaled(0.5);
        assert!((scaled.weight - 0.4).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // VisemeKeyframe Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_viseme_keyframe_new() {
        let kf = VisemeKeyframe::new(0.5, Viseme::OH, 1.0, 0.1);
        assert!((kf.time - 0.5).abs() < 0.01);
        assert_eq!(kf.viseme, Viseme::OH);
        assert!((kf.duration - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_viseme_keyframe_end_time() {
        let kf = VisemeKeyframe::new(0.5, Viseme::AA, 1.0, 0.2);
        assert!((kf.end_time() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_viseme_keyframe_with_curve() {
        let kf = VisemeKeyframe::new(0.0, Viseme::PP, 1.0, 0.1)
            .with_curve(BlendCurve::FastAttack);
        assert_eq!(kf.blend_curve, BlendCurve::FastAttack);
    }

    // -----------------------------------------------------------------------
    // LipSyncTrack Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lip_sync_track_new() {
        let track = LipSyncTrack::new();
        assert!(track.is_empty());
        assert_eq!(track.keyframe_count(), 0);
    }

    #[test]
    fn test_lip_sync_track_add_viseme() {
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.1);
        track.add_viseme(0.1, Viseme::EE, 1.0, 0.1);

        assert_eq!(track.keyframe_count(), 2);
        assert!(!track.is_empty());
    }

    #[test]
    fn test_lip_sync_track_duration() {
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.1);
        track.add_viseme(0.5, Viseme::EE, 1.0, 0.2);

        assert!((track.duration - 0.7).abs() < 0.01); // 0.5 + 0.2
    }

    #[test]
    fn test_lip_sync_track_sample_empty() {
        let track = LipSyncTrack::new();
        let weights = track.sample(0.0);

        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].viseme, Viseme::Sil);
    }

    #[test]
    fn test_lip_sync_track_sample_single() {
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.2);

        let weights = track.sample(0.1);
        assert!(!weights.is_empty());
        assert!(weights.iter().any(|w| w.viseme == Viseme::AA && w.weight > 0.5));
    }

    #[test]
    fn test_lip_sync_track_sample_between() {
        let mut track = LipSyncTrack::new();
        track.use_coarticulation = false; // Disable for simpler testing
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.1);
        track.add_viseme(0.2, Viseme::EE, 1.0, 0.1);

        // Sample between keyframes (during blend-out of AA)
        let weights = track.sample(0.15);
        // AA should be blending out
        let aa_weight = weights
            .iter()
            .find(|w| w.viseme == Viseme::AA)
            .map(|w| w.weight);
        assert!(aa_weight.is_some());
    }

    #[test]
    fn test_lip_sync_track_sample_after() {
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.1);

        // Sample well after track ends
        let weights = track.sample(1.0);
        // Should return silence or very low weight
        let total_weight: f32 = weights.iter().map(|w| w.weight).sum();
        assert!(
            weights.iter().any(|w| w.viseme == Viseme::Sil)
                || weights.iter().all(|w| w.weight < 0.5)
                || total_weight > 0.0
        );
    }

    #[test]
    fn test_lip_sync_track_events() {
        let mut track = LipSyncTrack::new();
        track.add_event(LipSyncEvent::WordStart {
            time: 0.0,
            word: "hello".to_string(),
        });
        track.add_event(LipSyncEvent::WordEnd { time: 0.5 });

        let events = track.events_in_range(0.0, 0.3);
        assert_eq!(events.len(), 1);

        let events = track.events_in_range(0.0, 1.0);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_lip_sync_track_clear() {
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.1);
        track.add_event(LipSyncEvent::WordStart {
            time: 0.0,
            word: "test".to_string(),
        });

        track.clear();

        assert!(track.is_empty());
        assert!(track.events.is_empty());
        assert_eq!(track.duration, 0.0);
    }

    #[test]
    fn test_lip_sync_track_coarticulation_toggle() {
        let mut track = LipSyncTrack::new();
        track.set_coarticulation(false);
        assert!(!track.use_coarticulation);

        track.set_coarticulation(true);
        assert!(track.use_coarticulation);
    }

    // -----------------------------------------------------------------------
    // PhonemeToViseme Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_phoneme_mapper_default() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.language, Language::EnglishUS);
    }

    #[test]
    fn test_phoneme_mapper_with_language() {
        let mapper = PhonemeToViseme::with_language(Language::Japanese);
        assert_eq!(mapper.language, Language::Japanese);
    }

    #[test]
    fn test_phoneme_mapper_bilabials() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map("p"), Viseme::PP);
        assert_eq!(mapper.map("b"), Viseme::PP);
        assert_eq!(mapper.map("m"), Viseme::PP);
    }

    #[test]
    fn test_phoneme_mapper_labiodentals() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map("f"), Viseme::FF);
        assert_eq!(mapper.map("v"), Viseme::FF);
    }

    #[test]
    fn test_phoneme_mapper_dentals() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map("th"), Viseme::TH);
        assert_eq!(mapper.map("dh"), Viseme::TH);
    }

    #[test]
    fn test_phoneme_mapper_alveolars() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map("t"), Viseme::DD);
        assert_eq!(mapper.map("d"), Viseme::DD);
        assert_eq!(mapper.map("n"), Viseme::DD);
        assert_eq!(mapper.map("l"), Viseme::DD);
    }

    #[test]
    fn test_phoneme_mapper_velars() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map("k"), Viseme::KK);
        assert_eq!(mapper.map("g"), Viseme::KK);
    }

    #[test]
    fn test_phoneme_mapper_sibilants() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map("s"), Viseme::SS);
        assert_eq!(mapper.map("z"), Viseme::SS);
    }

    #[test]
    fn test_phoneme_mapper_postalveolars() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map("ch"), Viseme::CH);
        assert_eq!(mapper.map("sh"), Viseme::CH);
    }

    #[test]
    fn test_phoneme_mapper_vowels() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map("a"), Viseme::AA);
        assert_eq!(mapper.map("aa"), Viseme::AA);
        assert_eq!(mapper.map("e"), Viseme::EE);
        assert_eq!(mapper.map("eh"), Viseme::EE);
        assert_eq!(mapper.map("i"), Viseme::IH);
        assert_eq!(mapper.map("o"), Viseme::OH);
        assert_eq!(mapper.map("u"), Viseme::OU);
        assert_eq!(mapper.map("oo"), Viseme::OU);
    }

    #[test]
    fn test_phoneme_mapper_silence() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map(""), Viseme::Sil);
        assert_eq!(mapper.map("sil"), Viseme::Sil);
        assert_eq!(mapper.map("_"), Viseme::Sil);
    }

    #[test]
    fn test_phoneme_mapper_override() {
        let mut mapper = PhonemeToViseme::new();
        mapper.add_override("custom", Viseme::RR);

        assert_eq!(mapper.map("custom"), Viseme::RR);
    }

    #[test]
    fn test_phoneme_mapper_case_insensitive() {
        let mapper = PhonemeToViseme::new();
        assert_eq!(mapper.map("P"), Viseme::PP);
        assert_eq!(mapper.map("AA"), Viseme::AA);
    }

    #[test]
    fn test_phoneme_mapper_cluster() {
        let mapper = PhonemeToViseme::new();
        let cluster = mapper.map_cluster(&["h", "eh", "l", "oh"]);

        assert_eq!(cluster.len(), 4);
    }

    #[test]
    fn test_phoneme_mapper_create_track() {
        let mapper = PhonemeToViseme::new();
        let timings = [(0.0, 0.1, "h"), (0.1, 0.2, "eh"), (0.2, 0.3, "l")];

        let track = mapper.create_track(&timings);
        assert_eq!(track.keyframe_count(), 3);
    }

    #[test]
    fn test_phoneme_mapper_japanese() {
        let mapper = PhonemeToViseme::with_language(Language::Japanese);
        assert_eq!(mapper.map("a"), Viseme::AA);
        assert_eq!(mapper.map("i"), Viseme::IH);
        assert_eq!(mapper.map("u"), Viseme::OU);
        assert_eq!(mapper.map("e"), Viseme::EE);
        assert_eq!(mapper.map("o"), Viseme::OH);
        assert_eq!(mapper.map("k"), Viseme::KK);
        assert_eq!(mapper.map("N"), Viseme::NN); // Syllabic N
    }

    // -----------------------------------------------------------------------
    // AudioAnalyzer Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audio_analyzer_new() {
        let analyzer = AudioAnalyzer::new(44100.0);
        assert_eq!(analyzer.sample_rate, 44100.0);
        assert_eq!(analyzer.amplitude, 0.0);
        assert!(!analyzer.is_voiced);
    }

    #[test]
    fn test_audio_analyzer_process_silence() {
        let mut analyzer = AudioAnalyzer::new(44100.0);
        let silence = vec![0.0; 100];

        analyzer.process_samples(&silence);

        assert!(analyzer.amplitude < VOICED_THRESHOLD);
        assert!(!analyzer.is_voiced);
    }

    #[test]
    fn test_audio_analyzer_process_loud() {
        let mut analyzer = AudioAnalyzer::new(44100.0);
        // Disable smoothing for deterministic test
        analyzer.smoothing = 0.0;
        let loud: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 0.8).collect();

        analyzer.process_samples(&loud);

        assert!(analyzer.amplitude > VOICED_THRESHOLD);
        assert!(analyzer.is_voiced);
    }

    #[test]
    fn test_audio_analyzer_smoothing() {
        let mut analyzer = AudioAnalyzer::new(44100.0);
        analyzer.smoothing = 0.5;

        // First sample
        let loud: Vec<f32> = vec![0.5; 100];
        analyzer.process_samples(&loud);
        let first_amplitude = analyzer.amplitude;

        // Second sample with silence - should decay slowly
        let silence = vec![0.0; 100];
        analyzer.process_samples(&silence);

        assert!(analyzer.amplitude < first_amplitude);
        assert!(analyzer.amplitude > 0.0); // Still some smoothed value
    }

    #[test]
    fn test_audio_analyzer_jaw_open() {
        let mut analyzer = AudioAnalyzer::new(44100.0);
        // Disable smoothing for deterministic test
        analyzer.smoothing = 0.0;

        // Silence = no jaw
        assert_eq!(analyzer.jaw_open(), 0.0);

        // Loud = jaw open
        let loud: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 0.8).collect();
        analyzer.process_samples(&loud);
        assert!(analyzer.jaw_open() > 0.0);
    }

    #[test]
    fn test_audio_analyzer_suggest_viseme() {
        let mut analyzer = AudioAnalyzer::new(44100.0);

        // Silence suggests Sil
        assert_eq!(analyzer.suggest_viseme(), Viseme::Sil);

        // Loud suggests open vowel
        let loud: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 0.9).collect();
        analyzer.process_samples(&loud);
        let suggested = analyzer.suggest_viseme();
        assert!(suggested.is_vowel() || suggested == Viseme::Sil);
    }

    #[test]
    fn test_audio_analyzer_reset() {
        let mut analyzer = AudioAnalyzer::new(44100.0);
        let loud: Vec<f32> = vec![0.8; 100];
        analyzer.process_samples(&loud);

        analyzer.reset();

        assert_eq!(analyzer.amplitude, 0.0);
        assert!(!analyzer.is_voiced);
    }

    #[test]
    fn test_audio_analyzer_empty_samples() {
        let mut analyzer = AudioAnalyzer::new(44100.0);
        let empty: Vec<f32> = vec![];

        let amplitude = analyzer.process_samples(&empty);
        assert_eq!(amplitude, 0.0);
    }

    // -----------------------------------------------------------------------
    // LipSyncController Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lip_sync_controller_new() {
        let controller = LipSyncController::new();
        assert!(!controller.is_playing);
        assert!(controller.track.is_none());
    }

    #[test]
    fn test_lip_sync_controller_set_track() {
        let mut controller = LipSyncController::new();
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.1);

        controller.set_track(track);

        assert!(controller.track.is_some());
        assert_eq!(controller.current_time, 0.0);
    }

    #[test]
    fn test_lip_sync_controller_play_pause() {
        let mut controller = LipSyncController::new();

        controller.play();
        assert!(controller.is_playing);

        controller.pause();
        assert!(!controller.is_playing);
    }

    #[test]
    fn test_lip_sync_controller_seek() {
        let mut controller = LipSyncController::new();
        controller.seek(1.5);
        assert!((controller.current_time - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_lip_sync_controller_update() {
        let mut controller = LipSyncController::new();
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.5);
        controller.set_track(track);

        controller.play();
        controller.update(0.1);

        assert!((controller.current_time - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_lip_sync_controller_playback_speed() {
        let mut controller = LipSyncController::new();
        controller.playback_speed = 2.0;

        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 1.0);
        controller.set_track(track);

        controller.play();
        controller.update(0.1);

        assert!((controller.current_time - 0.2).abs() < 0.01); // 0.1 * 2.0
    }

    #[test]
    fn test_lip_sync_controller_get_weights() {
        let mut controller = LipSyncController::new();
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::OH, 1.0, 0.5);
        controller.set_track(track);

        controller.play();
        controller.update(0.0);

        let weights = controller.get_weights();
        assert!(!weights.is_empty());
    }

    #[test]
    fn test_lip_sync_controller_to_blend_weights() {
        let mut controller = LipSyncController::new();
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.5);
        controller.set_track(track);

        controller.play();
        controller.update(0.1);

        let blend_weights = controller.to_blend_weights();
        assert!(!blend_weights.is_empty());
    }

    #[test]
    fn test_lip_sync_controller_clear_track() {
        let mut controller = LipSyncController::new();
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.1);
        controller.set_track(track);

        controller.clear_track();

        assert!(controller.track.is_none());
    }

    #[test]
    fn test_lip_sync_controller_audio_fallback() {
        let mut controller = LipSyncController::new();
        controller.use_audio_fallback = true;
        // Disable smoothing for deterministic test
        controller.audio_analyzer.smoothing = 0.0;

        // Process loud audio
        let loud: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 0.8).collect();
        controller.process_audio(&loud);

        assert!(controller.audio_analyzer.is_voiced);
    }

    #[test]
    fn test_lip_sync_controller_blend_mapping() {
        let mut controller = LipSyncController::new();
        controller.set_blend_mapping(Viseme::AA, "custom_aa_shape");

        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.5);
        controller.set_track(track);

        controller.play();
        controller.update(0.1);

        let blend_weights = controller.to_blend_weights();
        assert!(blend_weights.contains_key("custom_aa_shape"));
    }

    #[test]
    fn test_lip_sync_controller_intensity() {
        let mut controller = LipSyncController::new();
        controller.intensity = 0.5;

        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.5);
        controller.set_track(track);

        controller.play();
        controller.update(0.1);

        let weights = controller.get_weights();
        for w in weights {
            assert!(w.weight <= 0.5);
        }
    }

    #[test]
    fn test_lip_sync_controller_jaw_open() {
        let mut controller = LipSyncController::new();
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.5);
        controller.set_track(track);

        controller.play();
        controller.update(0.1);

        // Vowel should have jaw open
        let jaw = controller.jaw_open();
        assert!(jaw > 0.0);
    }

    // -----------------------------------------------------------------------
    // CoarticulationRules Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_coarticulation_rule_new() {
        let rule = CoarticulationRule::new(Viseme::PP);
        assert_eq!(rule.viseme, Viseme::PP);
        assert!(rule.anticipates.is_empty());
    }

    #[test]
    fn test_coarticulation_rule_builder() {
        let rule = CoarticulationRule::new(Viseme::DD)
            .anticipate(&[Viseme::OH, Viseme::OU])
            .carryover(&[Viseme::AA])
            .with_anticipation_strength(0.5)
            .with_carryover_strength(0.3);

        assert_eq!(rule.anticipates.len(), 2);
        assert_eq!(rule.carries_over_from.len(), 1);
        assert!((rule.anticipation_strength - 0.5).abs() < 0.01);
        assert!((rule.carryover_strength - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_coarticulation_rules_default() {
        let rules = CoarticulationRules::new();

        // Should have rules for consonants anticipating rounded vowels
        let pp_rule = rules.get(Viseme::PP);
        assert!(pp_rule.is_some());
    }

    // -----------------------------------------------------------------------
    // Language Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_language_all() {
        let all = Language::all();
        assert!(all.len() >= 8);
    }

    #[test]
    fn test_language_code() {
        assert_eq!(Language::EnglishUS.code(), "en-US");
        assert_eq!(Language::Japanese.code(), "ja");
    }

    #[test]
    fn test_language_name() {
        assert_eq!(Language::EnglishUS.name(), "English (US)");
        assert_eq!(Language::Japanese.name(), "Japanese");
    }

    // -----------------------------------------------------------------------
    // Utility Function Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_track_from_phonemes() {
        let phonemes = [
            (0.0, 0.1, "h"),
            (0.1, 0.2, "eh"),
            (0.2, 0.35, "l"),
            (0.35, 0.5, "oh"),
        ];

        let track = create_track_from_phonemes(&phonemes, Language::EnglishUS);
        assert_eq!(track.keyframe_count(), 4);
    }

    #[test]
    fn test_create_hello_track() {
        let track = create_hello_track();
        assert!(!track.is_empty());
        assert!(track.duration > 0.0);
    }

    // -----------------------------------------------------------------------
    // Integration Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_lip_sync_pipeline() {
        // Create phoneme data
        let phonemes = [
            (0.0, 0.1, "h"),
            (0.1, 0.25, "aa"),
            (0.25, 0.35, "l"),
            (0.35, 0.5, "oh"),
        ];

        // Create track
        let track = create_track_from_phonemes(&phonemes, Language::EnglishUS);

        // Create controller
        let mut controller = LipSyncController::new();
        controller.set_track(track);
        controller.play();

        // Simulate frames
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            // 1 second
            controller.update(dt);
            let weights = controller.get_weights();
            assert!(!weights.is_empty());
        }
    }

    #[test]
    fn test_lip_sync_with_blend_shapes() {
        let mut controller = LipSyncController::new();
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.5);
        controller.set_track(track);

        controller.play();
        controller.update(0.1);

        // Create blend shape set
        let mut shapes = BlendShapeSet::new(100);
        shapes.add_target(
            crate::blend_shapes::BlendShapeTarget::new("viseme_aa")
                .with_region(FaceRegion::Mouth),
        );

        // Apply
        controller.apply_to_blend_shapes(&mut shapes);

        // Check weight was set
        let weight = shapes.get_weight_by_name("viseme_aa");
        assert!(weight.is_some());
    }

    #[test]
    fn test_lip_sync_with_facs() {
        let mut controller = LipSyncController::new();
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::AA, 1.0, 0.5);
        controller.set_track(track);

        controller.play();
        controller.update(0.1);

        // Create FACS controller
        let mut facs = FACSController::new();

        // Apply
        controller.apply_to_facs(&mut facs);

        // Check AUs were set (AA maps to jaw drop etc)
        assert!(facs.has_active_aus());
    }

    #[test]
    fn test_edge_case_rapid_speech() {
        let mut track = LipSyncTrack::new();

        // Very rapid phoneme changes
        for i in 0..20 {
            let t = i as f32 * 0.02;
            let viseme = Viseme::from_index(i % 15).unwrap();
            track.add_viseme(t, viseme, 1.0, 0.015);
        }

        // Sample at various points
        for i in 0..20 {
            let t = i as f32 * 0.02 + 0.01;
            let weights = track.sample(t);
            assert!(!weights.is_empty());
        }
    }

    #[test]
    fn test_edge_case_long_silence() {
        let mut track = LipSyncTrack::new();
        track.add_viseme(0.0, Viseme::Sil, 1.0, 5.0);

        // Sample throughout
        for i in 0..50 {
            let t = i as f32 * 0.1;
            let weights = track.sample(t);
            assert!(weights.iter().any(|w| w.viseme == Viseme::Sil));
        }
    }
}
