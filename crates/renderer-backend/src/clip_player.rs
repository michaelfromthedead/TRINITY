//! Animation clip player for skeletal animation playback (T-AN-2.1).
//!
//! This module provides the ClipPlayer struct for controlling animation clip
//! playback with support for:
//!
//! - Play/pause/stop/resume controls
//! - Variable play rate (forward, reverse, time scaling)
//! - Multiple loop modes (once, loop, ping-pong)
//! - Seeking to arbitrary times
//! - Event firing when playback crosses event times
//! - Curve sampling for blend shapes and parameters
//!
//! # Architecture
//!
//! ```text
//! ClipPlayer
//! ├── AnimationClip (reference to clip data)
//! ├── PlaybackState (Stopped, Playing, Paused)
//! ├── LoopMode (Once, Loop, PingPong)
//! └── pending_events (fired events for current frame)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::clip_player::{ClipPlayer, PlaybackState};
//! use renderer_backend::animation_clip::AnimationClip;
//!
//! let clip = AnimationClip::new("walk", 1.0);
//! let mut player = ClipPlayer::new();
//! player.set_clip(clip);
//!
//! player.play();
//! assert_eq!(player.state(), PlaybackState::Playing);
//!
//! // Advance by delta time
//! player.update(0.016);
//!
//! // Get current pose
//! let pose = player.sample();
//!
//! // Check for fired events
//! for event in player.drain_events() {
//!     println!("Event fired: {}", event.name);
//! }
//! ```

use crate::animation_clip::{AnimationClip, AnimationEvent, CurveTrack, LoopMode, Pose};

// ---------------------------------------------------------------------------
// PlaybackState
// ---------------------------------------------------------------------------

/// The current state of animation playback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PlaybackState {
    /// Playback is stopped. Time is reset to 0.
    #[default]
    Stopped,

    /// Playback is active and time advances each update.
    Playing,

    /// Playback is paused. Time is frozen at current position.
    Paused,
}

impl PlaybackState {
    /// Returns true if the player is actively playing.
    #[inline]
    pub fn is_playing(&self) -> bool {
        matches!(self, Self::Playing)
    }

    /// Returns true if the player is paused.
    #[inline]
    pub fn is_paused(&self) -> bool {
        matches!(self, Self::Paused)
    }

    /// Returns true if the player is stopped.
    #[inline]
    pub fn is_stopped(&self) -> bool {
        matches!(self, Self::Stopped)
    }
}

impl std::fmt::Display for PlaybackState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => write!(f, "stopped"),
            Self::Playing => write!(f, "playing"),
            Self::Paused => write!(f, "paused"),
        }
    }
}

// ---------------------------------------------------------------------------
// ClipPlayer
// ---------------------------------------------------------------------------

/// Animation clip player for controlling playback.
///
/// ClipPlayer manages the playback state of an AnimationClip, including:
/// - Current time tracking with delta accumulation
/// - Play rate control (forward, reverse, time scaling)
/// - Loop mode handling (once, loop, ping-pong)
/// - Event detection and firing
/// - Pose sampling at current time
#[derive(Clone, Debug, Default)]
pub struct ClipPlayer {
    /// The animation clip being played.
    clip: Option<AnimationClip>,

    /// Current playback time in seconds.
    current_time: f32,

    /// Playback rate multiplier (1.0 = normal, 0.5 = half speed, -1.0 = reverse).
    play_rate: f32,

    /// How the animation loops.
    loop_mode: LoopMode,

    /// Current playback state.
    state: PlaybackState,

    /// Current playback direction for ping-pong mode.
    /// 1.0 = forward, -1.0 = reverse.
    direction: f32,

    /// Events that fired during the last update.
    pending_events: Vec<AnimationEvent>,

    /// Whether the animation has completed (for Once mode).
    completed: bool,
}

impl ClipPlayer {
    /// Create a new clip player with no clip.
    #[inline]
    pub fn new() -> Self {
        Self {
            clip: None,
            current_time: 0.0,
            play_rate: 1.0,
            loop_mode: LoopMode::Once,
            state: PlaybackState::Stopped,
            direction: 1.0,
            pending_events: Vec::new(),
            completed: false,
        }
    }

    /// Create a clip player with the given clip.
    #[inline]
    pub fn with_clip(clip: AnimationClip) -> Self {
        let loop_mode = clip.looping;
        Self {
            clip: Some(clip),
            current_time: 0.0,
            play_rate: 1.0,
            loop_mode,
            state: PlaybackState::Stopped,
            direction: 1.0,
            pending_events: Vec::new(),
            completed: false,
        }
    }

    // -------------------------------------------------------------------------
    // Clip Management
    // -------------------------------------------------------------------------

    /// Set the animation clip to play.
    ///
    /// This resets playback state to stopped and time to 0.
    pub fn set_clip(&mut self, clip: AnimationClip) {
        self.loop_mode = clip.looping;
        self.clip = Some(clip);
        self.reset();
    }

    /// Clear the current clip.
    pub fn clear_clip(&mut self) {
        self.clip = None;
        self.reset();
    }

    /// Get a reference to the current clip, if any.
    #[inline]
    pub fn clip(&self) -> Option<&AnimationClip> {
        self.clip.as_ref()
    }

    /// Get a mutable reference to the current clip, if any.
    #[inline]
    pub fn clip_mut(&mut self) -> Option<&mut AnimationClip> {
        self.clip.as_mut()
    }

    /// Check if a clip is loaded.
    #[inline]
    pub fn has_clip(&self) -> bool {
        self.clip.is_some()
    }

    // -------------------------------------------------------------------------
    // Playback Control
    // -------------------------------------------------------------------------

    /// Start or resume playback.
    ///
    /// If stopped, starts from the beginning.
    /// If paused, resumes from current position.
    pub fn play(&mut self) {
        if self.clip.is_none() {
            return;
        }

        match self.state {
            PlaybackState::Stopped => {
                self.current_time = if self.play_rate >= 0.0 {
                    0.0
                } else {
                    self.duration()
                };
                self.completed = false;
                self.state = PlaybackState::Playing;
            }
            PlaybackState::Paused => {
                self.state = PlaybackState::Playing;
            }
            PlaybackState::Playing => {
                // Already playing
            }
        }
    }

    /// Pause playback at the current position.
    pub fn pause(&mut self) {
        if self.state == PlaybackState::Playing {
            self.state = PlaybackState::Paused;
        }
    }

    /// Stop playback and reset to the beginning.
    pub fn stop(&mut self) {
        self.state = PlaybackState::Stopped;
        self.current_time = 0.0;
        self.direction = 1.0;
        self.completed = false;
        self.pending_events.clear();
    }

    /// Resume playback from paused state.
    ///
    /// Has no effect if not paused.
    pub fn resume(&mut self) {
        if self.state == PlaybackState::Paused {
            self.state = PlaybackState::Playing;
        }
    }

    /// Reset playback to the beginning without changing state.
    pub fn reset(&mut self) {
        self.current_time = 0.0;
        self.direction = 1.0;
        self.completed = false;
        self.pending_events.clear();
        self.state = PlaybackState::Stopped;
    }

    /// Toggle between playing and paused states.
    pub fn toggle_pause(&mut self) {
        match self.state {
            PlaybackState::Playing => self.pause(),
            PlaybackState::Paused => self.resume(),
            PlaybackState::Stopped => self.play(),
        }
    }

    // -------------------------------------------------------------------------
    // Playback State
    // -------------------------------------------------------------------------

    /// Get the current playback state.
    #[inline]
    pub fn state(&self) -> PlaybackState {
        self.state
    }

    /// Check if the player is currently playing.
    #[inline]
    pub fn is_playing(&self) -> bool {
        self.state.is_playing()
    }

    /// Check if the player is currently paused.
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.state.is_paused()
    }

    /// Check if the player is currently stopped.
    #[inline]
    pub fn is_stopped(&self) -> bool {
        self.state.is_stopped()
    }

    /// Check if the animation has completed (for Once mode).
    #[inline]
    pub fn is_completed(&self) -> bool {
        self.completed
    }

    // -------------------------------------------------------------------------
    // Time Control
    // -------------------------------------------------------------------------

    /// Get the current playback time in seconds.
    #[inline]
    pub fn current_time(&self) -> f32 {
        self.current_time
    }

    /// Set the current playback time directly.
    ///
    /// Time is clamped to [0, duration] or wrapped based on loop mode.
    pub fn set_time(&mut self, time: f32) {
        let duration = self.duration();
        if duration <= 0.0 {
            self.current_time = 0.0;
            return;
        }

        self.current_time = match self.loop_mode {
            LoopMode::Once => time.clamp(0.0, duration),
            LoopMode::Loop => time.rem_euclid(duration),
            LoopMode::PingPong => {
                let cycle_time = time.rem_euclid(duration * 2.0);
                if cycle_time <= duration {
                    self.direction = 1.0;
                    cycle_time
                } else {
                    self.direction = -1.0;
                    duration * 2.0 - cycle_time
                }
            }
        };
    }

    /// Seek to a specific time.
    ///
    /// This is an alias for `set_time` that also detects events.
    pub fn seek(&mut self, time: f32) {
        let old_time = self.current_time;
        self.set_time(time);

        // Fire events in the seek range
        if let Some(clip) = &self.clip {
            let (start, end) = if time >= old_time {
                (old_time, time)
            } else {
                (time, old_time)
            };

            for event in clip.events_in_range(start, end) {
                self.pending_events.push(event.clone());
            }
        }
    }

    /// Seek to a normalized position [0, 1].
    pub fn seek_normalized(&mut self, normalized: f32) {
        let duration = self.duration();
        self.seek(normalized.clamp(0.0, 1.0) * duration);
    }

    /// Get the current time as a normalized value [0, 1].
    #[inline]
    pub fn normalized_time(&self) -> f32 {
        let duration = self.duration();
        if duration <= 0.0 {
            0.0
        } else {
            self.current_time / duration
        }
    }

    /// Get the clip duration in seconds.
    #[inline]
    pub fn duration(&self) -> f32 {
        self.clip.as_ref().map_or(0.0, |c| c.duration)
    }

    /// Get the remaining time until the end (for Once mode).
    #[inline]
    pub fn remaining_time(&self) -> f32 {
        (self.duration() - self.current_time).max(0.0)
    }

    // -------------------------------------------------------------------------
    // Play Rate Control
    // -------------------------------------------------------------------------

    /// Get the current play rate.
    #[inline]
    pub fn play_rate(&self) -> f32 {
        self.play_rate
    }

    /// Set the play rate.
    ///
    /// - 1.0 = normal speed
    /// - 0.5 = half speed
    /// - 2.0 = double speed
    /// - -1.0 = reverse
    /// - 0.0 = paused (time won't advance)
    pub fn set_play_rate(&mut self, rate: f32) {
        self.play_rate = rate;
    }

    /// Check if playing in reverse (negative play rate).
    #[inline]
    pub fn is_reverse(&self) -> bool {
        self.play_rate < 0.0
    }

    /// Get the effective play rate considering ping-pong direction.
    #[inline]
    pub fn effective_play_rate(&self) -> f32 {
        self.play_rate * self.direction
    }

    // -------------------------------------------------------------------------
    // Loop Mode
    // -------------------------------------------------------------------------

    /// Get the current loop mode.
    #[inline]
    pub fn loop_mode(&self) -> LoopMode {
        self.loop_mode
    }

    /// Set the loop mode.
    pub fn set_loop_mode(&mut self, mode: LoopMode) {
        self.loop_mode = mode;
    }

    /// Set to play once and stop.
    #[inline]
    pub fn set_once(&mut self) {
        self.loop_mode = LoopMode::Once;
    }

    /// Set to loop continuously.
    #[inline]
    pub fn set_looping(&mut self) {
        self.loop_mode = LoopMode::Loop;
    }

    /// Set to ping-pong (forward then reverse).
    #[inline]
    pub fn set_ping_pong(&mut self) {
        self.loop_mode = LoopMode::PingPong;
    }

    // -------------------------------------------------------------------------
    // Update
    // -------------------------------------------------------------------------

    /// Update the playback by the given delta time.
    ///
    /// This advances (or reverses) the current time based on play rate,
    /// handles loop wrapping, and detects events that fired during this frame.
    pub fn update(&mut self, delta_time: f32) {
        if self.state != PlaybackState::Playing {
            return;
        }

        if self.clip.is_none() {
            return;
        }

        let duration = self.duration();
        if duration <= 0.0 {
            self.completed = true;
            self.state = PlaybackState::Stopped;
            return;
        }

        let old_time = self.current_time;
        let effective_rate = self.effective_play_rate();
        let new_time = self.current_time + delta_time * effective_rate;

        // Detect events before updating time
        self.detect_events(old_time, new_time, duration);

        // Update time based on loop mode
        match self.loop_mode {
            LoopMode::Once => {
                if effective_rate >= 0.0 {
                    if new_time >= duration {
                        self.current_time = duration;
                        self.completed = true;
                        self.state = PlaybackState::Stopped;
                    } else {
                        self.current_time = new_time;
                    }
                } else {
                    if new_time <= 0.0 {
                        self.current_time = 0.0;
                        self.completed = true;
                        self.state = PlaybackState::Stopped;
                    } else {
                        self.current_time = new_time;
                    }
                }
            }
            LoopMode::Loop => {
                self.current_time = new_time.rem_euclid(duration);
            }
            LoopMode::PingPong => {
                self.update_ping_pong(new_time, duration);
            }
        }
    }

    /// Update ping-pong mode with bounce handling.
    fn update_ping_pong(&mut self, new_time: f32, duration: f32) {
        let mut time = new_time;
        let mut dir = self.direction;

        // Handle bounces
        while time < 0.0 || time > duration {
            if time > duration {
                time = duration * 2.0 - time;
                dir = -dir;
            } else if time < 0.0 {
                time = -time;
                dir = -dir;
            }
        }

        self.current_time = time;
        self.direction = dir;
    }

    /// Detect and collect events that fire between old_time and new_time.
    ///
    /// Event detection uses semi-open intervals to avoid double-firing:
    /// - Forward: [old_time, new_time)
    /// - Reverse: (new_time, old_time]
    ///
    /// When wrapping occurs (loop mode), we detect events in both segments.
    fn detect_events(&mut self, old_time: f32, new_time: f32, duration: f32) {
        let clip = match &self.clip {
            Some(c) => c,
            None => return,
        };

        let effective_rate = self.effective_play_rate();
        let mut collected_events = Vec::new();

        // Handle forward playback
        if effective_rate >= 0.0 {
            if new_time >= old_time {
                // Normal forward progression: [old_time, new_time)
                for track in &clip.event_tracks {
                    for event in &track.events {
                        if event.time >= old_time && event.time < new_time {
                            collected_events.push(event.clone());
                        }
                    }
                }
            } else {
                // Wrapped around (loop mode): [old_time, duration] then [0, new_time)
                for track in &clip.event_tracks {
                    for event in &track.events {
                        // Include events from old_time to end of clip (inclusive)
                        if event.time >= old_time && event.time <= duration {
                            collected_events.push(event.clone());
                        }
                        // Include events from start to new_time (exclusive)
                        else if event.time >= 0.0 && event.time < new_time {
                            collected_events.push(event.clone());
                        }
                    }
                }
            }
        } else {
            // Handle reverse playback
            if new_time <= old_time {
                // Normal reverse progression: (new_time, old_time]
                // Collect events in the range, then reverse for proper order
                let mut events = Vec::new();
                for track in &clip.event_tracks {
                    for event in &track.events {
                        // Include old_time (<=), exclude new_time (<)
                        if event.time > new_time && event.time <= old_time {
                            events.push(event.clone());
                        }
                    }
                }
                events.sort_by(|a, b| b.time.partial_cmp(&a.time).unwrap_or(std::cmp::Ordering::Equal));
                collected_events.extend(events);
            } else {
                // Wrapped around in reverse (loop mode)
                // Go from old_time down to 0, then from duration down to new_time
                let mut events = Vec::new();
                for track in &clip.event_tracks {
                    for event in &track.events {
                        // Part 1: from old_time down to 0 (inclusive both ends)
                        if event.time >= 0.0 && event.time <= old_time {
                            events.push(event.clone());
                        }
                    }
                }
                events.sort_by(|a, b| b.time.partial_cmp(&a.time).unwrap_or(std::cmp::Ordering::Equal));
                collected_events.extend(events);

                let mut events2 = Vec::new();
                for track in &clip.event_tracks {
                    for event in &track.events {
                        // Part 2: from duration down to new_time (include duration, exclude new_time)
                        if event.time > new_time && event.time <= duration {
                            events2.push(event.clone());
                        }
                    }
                }
                events2.sort_by(|a, b| b.time.partial_cmp(&a.time).unwrap_or(std::cmp::Ordering::Equal));
                collected_events.extend(events2);
            }
        }

        self.pending_events.extend(collected_events);
    }

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    /// Get pending events that fired during the last update.
    #[inline]
    pub fn pending_events(&self) -> &[AnimationEvent] {
        &self.pending_events
    }

    /// Drain and return all pending events.
    pub fn drain_events(&mut self) -> Vec<AnimationEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Clear all pending events without returning them.
    pub fn clear_events(&mut self) {
        self.pending_events.clear();
    }

    /// Check if there are any pending events.
    #[inline]
    pub fn has_pending_events(&self) -> bool {
        !self.pending_events.is_empty()
    }

    // -------------------------------------------------------------------------
    // Sampling
    // -------------------------------------------------------------------------

    /// Sample the animation at the current time.
    ///
    /// Returns an empty pose if no clip is loaded.
    pub fn sample(&self) -> Pose {
        match &self.clip {
            Some(clip) => clip.sample(self.current_time),
            None => Pose::new(),
        }
    }

    /// Sample a specific bone at the current time.
    pub fn sample_bone(&self, bone_name: &str) -> Option<crate::skeleton::Transform> {
        self.clip.as_ref()?.sample_bone(bone_name, self.current_time)
    }

    /// Sample a curve at the current time.
    pub fn sample_curve(&self, curve_name: &str) -> Option<f32> {
        self.clip.as_ref()?.sample_curve(curve_name, self.current_time)
    }

    /// Sample all curves at the current time.
    pub fn sample_all_curves(&self) -> Vec<(String, f32)> {
        match &self.clip {
            Some(clip) => clip
                .curve_tracks
                .iter()
                .filter_map(|c| {
                    c.sample(self.current_time)
                        .map(|v| (c.name.clone(), v))
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get direct access to curve tracks.
    pub fn curve_tracks(&self) -> &[CurveTrack] {
        match &self.clip {
            Some(clip) => &clip.curve_tracks,
            None => &[],
        }
    }

    // -------------------------------------------------------------------------
    // Frame Info
    // -------------------------------------------------------------------------

    /// Get the current frame number (based on clip frame rate).
    pub fn current_frame(&self) -> u32 {
        match &self.clip {
            Some(clip) => clip.time_to_frame(self.current_time),
            None => 0,
        }
    }

    /// Get the total number of frames in the clip.
    pub fn frame_count(&self) -> u32 {
        match &self.clip {
            Some(clip) => clip.frame_count(),
            None => 0,
        }
    }

    /// Seek to a specific frame.
    pub fn seek_frame(&mut self, frame: u32) {
        if let Some(clip) = &self.clip {
            let time = clip.frame_to_time(frame);
            self.seek(time);
        }
    }

    /// Advance by one frame.
    pub fn next_frame(&mut self) {
        let frame = self.current_frame().saturating_add(1);
        self.seek_frame(frame);
    }

    /// Go back by one frame.
    pub fn prev_frame(&mut self) {
        let frame = self.current_frame().saturating_sub(1);
        self.seek_frame(frame);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation_clip::{
        AnimationClip, AnimationEvent, BoneTrack, CurveTrack, EventTrack,
        Keyframe, LoopMode, Track,
    };
    use glam::Vec3;

    // =========================================================================
    // PlaybackState Tests
    // =========================================================================

    #[test]
    fn test_playback_state_default() {
        assert_eq!(PlaybackState::default(), PlaybackState::Stopped);
    }

    #[test]
    fn test_playback_state_is_playing() {
        assert!(PlaybackState::Playing.is_playing());
        assert!(!PlaybackState::Paused.is_playing());
        assert!(!PlaybackState::Stopped.is_playing());
    }

    #[test]
    fn test_playback_state_is_paused() {
        assert!(!PlaybackState::Playing.is_paused());
        assert!(PlaybackState::Paused.is_paused());
        assert!(!PlaybackState::Stopped.is_paused());
    }

    #[test]
    fn test_playback_state_is_stopped() {
        assert!(!PlaybackState::Playing.is_stopped());
        assert!(!PlaybackState::Paused.is_stopped());
        assert!(PlaybackState::Stopped.is_stopped());
    }

    #[test]
    fn test_playback_state_display() {
        assert_eq!(format!("{}", PlaybackState::Stopped), "stopped");
        assert_eq!(format!("{}", PlaybackState::Playing), "playing");
        assert_eq!(format!("{}", PlaybackState::Paused), "paused");
    }

    // =========================================================================
    // ClipPlayer Construction Tests
    // =========================================================================

    #[test]
    fn test_clip_player_new() {
        let player = ClipPlayer::new();
        assert!(player.clip.is_none());
        assert_eq!(player.current_time, 0.0);
        assert_eq!(player.play_rate, 1.0);
        assert_eq!(player.loop_mode, LoopMode::Once);
        assert_eq!(player.state, PlaybackState::Stopped);
        assert_eq!(player.direction, 1.0);
        assert!(player.pending_events.is_empty());
    }

    #[test]
    fn test_clip_player_with_clip() {
        let clip = AnimationClip::looping("walk", 1.0);
        let player = ClipPlayer::with_clip(clip);

        assert!(player.clip.is_some());
        assert_eq!(player.loop_mode, LoopMode::Loop);
        assert_eq!(player.state, PlaybackState::Stopped);
    }

    #[test]
    fn test_clip_player_default() {
        let player = ClipPlayer::default();
        assert!(player.clip.is_none());
        assert_eq!(player.state, PlaybackState::Stopped);
    }

    // =========================================================================
    // Clip Management Tests
    // =========================================================================

    #[test]
    fn test_set_clip() {
        let mut player = ClipPlayer::new();
        let clip = AnimationClip::looping("run", 2.0);

        player.set_clip(clip);

        assert!(player.has_clip());
        assert_eq!(player.duration(), 2.0);
        assert_eq!(player.loop_mode, LoopMode::Loop);
    }

    #[test]
    fn test_set_clip_resets_state() {
        let mut player = ClipPlayer::new();
        player.set_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.5);

        // Setting a new clip should reset
        player.set_clip(AnimationClip::new("test2", 2.0));
        assert_eq!(player.state, PlaybackState::Stopped);
        assert_eq!(player.current_time, 0.0);
    }

    #[test]
    fn test_clear_clip() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();

        player.clear_clip();

        assert!(!player.has_clip());
        assert_eq!(player.state, PlaybackState::Stopped);
    }

    #[test]
    fn test_clip_accessor() {
        let player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        assert!(player.clip().is_some());
        assert_eq!(player.clip().unwrap().name, "test");
    }

    #[test]
    fn test_clip_mut_accessor() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        if let Some(clip) = player.clip_mut() {
            clip.name = "modified".to_string();
        }
        assert_eq!(player.clip().unwrap().name, "modified");
    }

    // =========================================================================
    // Play/Pause/Stop State Transition Tests
    // =========================================================================

    #[test]
    fn test_play_from_stopped() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        assert_eq!(player.state(), PlaybackState::Stopped);

        player.play();

        assert_eq!(player.state(), PlaybackState::Playing);
        assert_eq!(player.current_time(), 0.0);
    }

    #[test]
    fn test_play_without_clip_does_nothing() {
        let mut player = ClipPlayer::new();
        player.play();
        assert_eq!(player.state(), PlaybackState::Stopped);
    }

    #[test]
    fn test_pause_from_playing() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.3);

        player.pause();

        assert_eq!(player.state(), PlaybackState::Paused);
        assert!((player.current_time() - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_pause_when_not_playing() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        // Pausing when stopped should have no effect
        player.pause();
        assert_eq!(player.state(), PlaybackState::Stopped);
    }

    #[test]
    fn test_resume_from_paused() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.3);
        player.pause();

        let time_before = player.current_time();
        player.resume();

        assert_eq!(player.state(), PlaybackState::Playing);
        assert_eq!(player.current_time(), time_before);
    }

    #[test]
    fn test_resume_when_not_paused() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        // Resume when stopped should have no effect
        player.resume();
        assert_eq!(player.state(), PlaybackState::Stopped);

        // Resume when playing should have no effect
        player.play();
        player.resume();
        assert_eq!(player.state(), PlaybackState::Playing);
    }

    #[test]
    fn test_stop_from_playing() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.5);

        player.stop();

        assert_eq!(player.state(), PlaybackState::Stopped);
        assert_eq!(player.current_time(), 0.0);
    }

    #[test]
    fn test_stop_from_paused() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.5);
        player.pause();

        player.stop();

        assert_eq!(player.state(), PlaybackState::Stopped);
        assert_eq!(player.current_time(), 0.0);
    }

    #[test]
    fn test_toggle_pause_from_stopped() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.toggle_pause();
        assert_eq!(player.state(), PlaybackState::Playing);
    }

    #[test]
    fn test_toggle_pause_from_playing() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.toggle_pause();
        assert_eq!(player.state(), PlaybackState::Paused);
    }

    #[test]
    fn test_toggle_pause_from_paused() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.pause();
        player.toggle_pause();
        assert_eq!(player.state(), PlaybackState::Playing);
    }

    #[test]
    fn test_reset() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.5);

        player.reset();

        assert_eq!(player.state(), PlaybackState::Stopped);
        assert_eq!(player.current_time(), 0.0);
        assert_eq!(player.direction, 1.0);
        assert!(!player.is_completed());
    }

    // =========================================================================
    // Forward/Reverse Playback Tests
    // =========================================================================

    #[test]
    fn test_forward_playback() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();

        player.update(0.25);
        assert!((player.current_time() - 0.25).abs() < 1e-5);

        player.update(0.25);
        assert!((player.current_time() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_reverse_playback() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_play_rate(-1.0);
        player.play();

        // When starting in reverse from stopped, should start at duration
        assert!((player.current_time() - 1.0).abs() < 1e-5);

        player.update(0.25);
        assert!((player.current_time() - 0.75).abs() < 1e-5);

        player.update(0.5);
        assert!((player.current_time() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_reverse_playback_completion() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_play_rate(-1.0);
        player.play();

        player.update(1.5); // Should reach 0 and stop

        assert_eq!(player.current_time(), 0.0);
        assert!(player.is_completed());
        assert_eq!(player.state(), PlaybackState::Stopped);
    }

    #[test]
    fn test_is_reverse() {
        let mut player = ClipPlayer::new();
        assert!(!player.is_reverse());

        player.set_play_rate(-1.0);
        assert!(player.is_reverse());

        player.set_play_rate(0.0);
        assert!(!player.is_reverse());
    }

    // =========================================================================
    // Time Scaling Tests
    // =========================================================================

    #[test]
    fn test_half_speed_playback() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_play_rate(0.5);
        player.play();

        player.update(0.5); // Should advance by 0.25
        assert!((player.current_time() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_double_speed_playback() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_play_rate(2.0);
        player.play();

        player.update(0.25); // Should advance by 0.5
        assert!((player.current_time() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_zero_play_rate() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_play_rate(0.0);
        player.play();

        player.update(1.0);
        assert_eq!(player.current_time(), 0.0); // Time should not advance
    }

    #[test]
    fn test_negative_play_rate_from_middle() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.5);

        // Now reverse
        player.set_play_rate(-1.0);
        player.update(0.25);

        assert!((player.current_time() - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_play_rate_accessor() {
        let mut player = ClipPlayer::new();
        assert_eq!(player.play_rate(), 1.0);

        player.set_play_rate(2.5);
        assert_eq!(player.play_rate(), 2.5);
    }

    #[test]
    fn test_effective_play_rate() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_play_rate(2.0);

        assert_eq!(player.effective_play_rate(), 2.0);

        // In ping-pong mode with reversed direction
        player.direction = -1.0;
        assert_eq!(player.effective_play_rate(), -2.0);
    }

    // =========================================================================
    // Loop Mode Tests
    // =========================================================================

    #[test]
    fn test_loop_mode_once_stops_at_end() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::Once);
        player.play();

        player.update(1.5); // Go past end

        assert_eq!(player.current_time(), 1.0);
        assert!(player.is_completed());
        assert_eq!(player.state(), PlaybackState::Stopped);
    }

    #[test]
    fn test_loop_mode_loop_wraps() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::Loop);
        player.play();

        player.update(1.25); // Go past end

        assert!((player.current_time() - 0.25).abs() < 1e-5);
        assert!(!player.is_completed());
        assert_eq!(player.state(), PlaybackState::Playing);
    }

    #[test]
    fn test_loop_mode_loop_multiple_wraps() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::Loop);
        player.play();

        player.update(3.5); // 3.5 loops

        assert!((player.current_time() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_loop_mode_ping_pong_reverses() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::PingPong);
        player.play();

        player.update(1.5); // Go to 1.5, should bounce back to 0.5

        assert!((player.current_time() - 0.5).abs() < 1e-5);
        assert_eq!(player.direction, -1.0); // Should be reversed
    }

    #[test]
    fn test_loop_mode_ping_pong_multiple_bounces() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::PingPong);
        player.play();

        // Forward to 1.0, back to 0.0, forward again
        player.update(2.5);

        assert!((player.current_time() - 0.5).abs() < 1e-5);
        assert_eq!(player.direction, 1.0); // Should be forward again
    }

    #[test]
    fn test_set_loop_mode_helpers() {
        let mut player = ClipPlayer::new();

        player.set_once();
        assert_eq!(player.loop_mode(), LoopMode::Once);

        player.set_looping();
        assert_eq!(player.loop_mode(), LoopMode::Loop);

        player.set_ping_pong();
        assert_eq!(player.loop_mode(), LoopMode::PingPong);
    }

    // =========================================================================
    // Seek Tests
    // =========================================================================

    #[test]
    fn test_set_time() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        player.set_time(0.5);
        assert_eq!(player.current_time(), 0.5);
    }

    #[test]
    fn test_set_time_clamps_once_mode() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::Once);

        player.set_time(1.5);
        assert_eq!(player.current_time(), 1.0);

        player.set_time(-0.5);
        assert_eq!(player.current_time(), 0.0);
    }

    #[test]
    fn test_set_time_wraps_loop_mode() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::Loop);

        player.set_time(1.5);
        assert!((player.current_time() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_set_time_handles_ping_pong() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::PingPong);

        player.set_time(1.5);
        assert!((player.current_time() - 0.5).abs() < 1e-5);
        assert_eq!(player.direction, -1.0);
    }

    #[test]
    fn test_seek() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.seek(0.75);
        assert!((player.current_time() - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_seek_normalized() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 2.0));
        player.seek_normalized(0.5);
        assert!((player.current_time() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_seek_normalized_clamps() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        player.seek_normalized(1.5);
        assert_eq!(player.current_time(), 1.0);

        player.seek_normalized(-0.5);
        assert_eq!(player.current_time(), 0.0);
    }

    #[test]
    fn test_normalized_time() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 2.0));
        player.set_time(1.0);
        assert!((player.normalized_time() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_seek_beyond_bounds() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::Once);

        player.seek(10.0);
        assert_eq!(player.current_time(), 1.0);

        player.seek(-5.0);
        assert_eq!(player.current_time(), 0.0);
    }

    // =========================================================================
    // Event Firing Tests
    // =========================================================================

    fn create_clip_with_events() -> AnimationClip {
        let mut clip = AnimationClip::new("test", 1.0);
        let mut event_track = EventTrack::new("notifies");
        event_track.add_event(AnimationEvent::new("start", 0.0));
        event_track.add_event(AnimationEvent::new("quarter", 0.25));
        event_track.add_event(AnimationEvent::new("half", 0.5));
        event_track.add_event(AnimationEvent::new("three_quarter", 0.75));
        event_track.add_event(AnimationEvent::new("end", 1.0));
        clip.add_event_track(event_track);
        clip
    }

    #[test]
    fn test_event_firing_forward() {
        let mut player = ClipPlayer::with_clip(create_clip_with_events());
        player.play();

        // Update from 0 to 0.3 should fire "start" and "quarter"
        player.update(0.3);
        let events = player.drain_events();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].name, "start");
        assert_eq!(events[1].name, "quarter");
    }

    #[test]
    fn test_event_firing_multiple_updates() {
        let mut player = ClipPlayer::with_clip(create_clip_with_events());
        player.play();

        player.update(0.3);
        let events1 = player.drain_events();
        assert_eq!(events1.len(), 2);

        player.update(0.3); // 0.3 to 0.6, should fire "half"
        let events2 = player.drain_events();
        assert_eq!(events2.len(), 1);
        assert_eq!(events2[0].name, "half");
    }

    #[test]
    fn test_event_firing_reverse() {
        let mut player = ClipPlayer::with_clip(create_clip_with_events());
        player.set_play_rate(-1.0);
        player.play();

        // Starts at 1.0, update to 0.7 should fire "end" and "three_quarter"
        // Events should be in reverse time order (highest time first)
        player.update(0.3);
        let events = player.drain_events();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].name, "end");         // at 1.0
        assert_eq!(events[1].name, "three_quarter"); // at 0.75
    }

    #[test]
    fn test_event_firing_loop_wrap() {
        let mut player = ClipPlayer::with_clip(create_clip_with_events());
        player.set_loop_mode(LoopMode::Loop);
        player.play();

        player.update(0.8); // 0 to 0.8
        player.drain_events();

        // Now from 0.8 to 1.1 (wraps to 0.1) should fire "end" and "start"
        player.update(0.3);
        let events = player.drain_events();

        // Should have "end" (at 1.0) and "start" (at 0.0)
        let event_names: Vec<_> = events.iter().map(|e| e.name.as_str()).collect();
        assert!(event_names.contains(&"end"));
        assert!(event_names.contains(&"start"));
    }

    #[test]
    fn test_drain_events_clears_pending() {
        let mut player = ClipPlayer::with_clip(create_clip_with_events());
        player.play();
        player.update(0.3);

        assert!(player.has_pending_events());
        let _ = player.drain_events();
        assert!(!player.has_pending_events());
    }

    #[test]
    fn test_clear_events() {
        let mut player = ClipPlayer::with_clip(create_clip_with_events());
        player.play();
        player.update(0.3);

        player.clear_events();
        assert!(player.pending_events().is_empty());
    }

    #[test]
    fn test_pending_events_accessor() {
        let mut player = ClipPlayer::with_clip(create_clip_with_events());
        player.play();
        player.update(0.3);

        let events = player.pending_events();
        assert_eq!(events.len(), 2);
    }

    // =========================================================================
    // Curve Sampling Tests
    // =========================================================================

    fn create_clip_with_curves() -> AnimationClip {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.add_curve_track(CurveTrack::with_keyframes(
            "alpha",
            vec![
                Keyframe::linear(0.0, 0.0),
                Keyframe::linear(1.0, 1.0),
            ],
        ));
        clip.add_curve_track(CurveTrack::with_keyframes(
            "intensity",
            vec![
                Keyframe::linear(0.0, 1.0),
                Keyframe::linear(0.5, 2.0),
                Keyframe::linear(1.0, 1.0),
            ],
        ));
        clip
    }

    #[test]
    fn test_sample_curve() {
        let mut player = ClipPlayer::with_clip(create_clip_with_curves());
        player.set_time(0.5);

        let alpha = player.sample_curve("alpha").unwrap();
        assert!((alpha - 0.5).abs() < 1e-5);

        let intensity = player.sample_curve("intensity").unwrap();
        assert!((intensity - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_curve_not_found() {
        let player = ClipPlayer::with_clip(create_clip_with_curves());
        assert!(player.sample_curve("nonexistent").is_none());
    }

    #[test]
    fn test_sample_all_curves() {
        let mut player = ClipPlayer::with_clip(create_clip_with_curves());
        player.set_time(0.5);

        let curves = player.sample_all_curves();
        assert_eq!(curves.len(), 2);

        let alpha = curves.iter().find(|(n, _)| n == "alpha").unwrap().1;
        assert!((alpha - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_curve_tracks_accessor() {
        let player = ClipPlayer::with_clip(create_clip_with_curves());
        let tracks = player.curve_tracks();
        assert_eq!(tracks.len(), 2);
    }

    #[test]
    fn test_curve_sampling_accuracy() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.add_curve_track(CurveTrack::with_keyframes(
            "linear",
            vec![
                Keyframe::linear(0.0, 0.0),
                Keyframe::linear(1.0, 100.0),
            ],
        ));

        let player = ClipPlayer::with_clip(clip);

        // Test multiple points
        for i in 0..=10 {
            let t = i as f32 * 0.1;
            let mut p = player.clone();
            p.set_time(t);
            let value = p.sample_curve("linear").unwrap();
            let expected = t * 100.0;
            assert!(
                (value - expected).abs() < 1e-4,
                "At t={}, expected {} but got {}",
                t,
                expected,
                value
            );
        }
    }

    // =========================================================================
    // Pose Sampling Tests
    // =========================================================================

    fn create_clip_with_bones() -> AnimationClip {
        let mut clip = AnimationClip::new("test", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);

        clip.add_bone_track(BoneTrack::new("hip").with_position(pos_track));
        clip
    }

    #[test]
    fn test_sample() {
        let mut player = ClipPlayer::with_clip(create_clip_with_bones());
        player.set_time(0.5);

        let pose = player.sample();
        let hip = pose.get("hip").unwrap();
        assert!(hip.position.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_sample_without_clip() {
        let player = ClipPlayer::new();
        let pose = player.sample();
        assert!(pose.is_empty());
    }

    #[test]
    fn test_sample_bone() {
        let mut player = ClipPlayer::with_clip(create_clip_with_bones());
        player.set_time(0.25);

        let transform = player.sample_bone("hip").unwrap();
        assert!(transform.position.abs_diff_eq(Vec3::new(2.5, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_sample_bone_not_found() {
        let player = ClipPlayer::with_clip(create_clip_with_bones());
        assert!(player.sample_bone("nonexistent").is_none());
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_zero_duration_clip() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("empty", 0.0));
        player.play();

        player.update(0.1);

        // Should complete immediately
        assert!(player.is_completed());
        assert_eq!(player.state(), PlaybackState::Stopped);
    }

    #[test]
    fn test_empty_clip() {
        let player = ClipPlayer::with_clip(AnimationClip::new("empty", 1.0));
        let pose = player.sample();
        assert!(pose.is_empty());
    }

    #[test]
    fn test_update_when_paused() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.3);
        player.pause();

        let time_before = player.current_time();
        player.update(0.5);

        assert_eq!(player.current_time(), time_before);
    }

    #[test]
    fn test_update_when_stopped() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        player.update(0.5);

        assert_eq!(player.current_time(), 0.0);
    }

    #[test]
    fn test_update_without_clip() {
        let mut player = ClipPlayer::new();
        player.state = PlaybackState::Playing;

        player.update(0.5);

        assert_eq!(player.current_time(), 0.0);
    }

    #[test]
    fn test_set_time_zero_duration() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("empty", 0.0));

        player.set_time(1.0);
        assert_eq!(player.current_time(), 0.0);
    }

    #[test]
    fn test_normalized_time_zero_duration() {
        let player = ClipPlayer::with_clip(AnimationClip::new("empty", 0.0));
        assert_eq!(player.normalized_time(), 0.0);
    }

    // =========================================================================
    // Time Accessors Tests
    // =========================================================================

    #[test]
    fn test_duration() {
        let player = ClipPlayer::with_clip(AnimationClip::new("test", 2.5));
        assert_eq!(player.duration(), 2.5);
    }

    #[test]
    fn test_duration_no_clip() {
        let player = ClipPlayer::new();
        assert_eq!(player.duration(), 0.0);
    }

    #[test]
    fn test_remaining_time() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_time(0.3);
        assert!((player.remaining_time() - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_remaining_time_past_end() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::Once);
        player.set_time(1.5);
        assert_eq!(player.remaining_time(), 0.0);
    }

    // =========================================================================
    // Frame Navigation Tests
    // =========================================================================

    #[test]
    fn test_current_frame() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.frame_rate = 30.0;
        let mut player = ClipPlayer::with_clip(clip);

        player.set_time(0.5);
        assert_eq!(player.current_frame(), 15);
    }

    #[test]
    fn test_frame_count() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.frame_rate = 30.0;
        let player = ClipPlayer::with_clip(clip);

        assert_eq!(player.frame_count(), 30);
    }

    #[test]
    fn test_seek_frame() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.frame_rate = 30.0;
        let mut player = ClipPlayer::with_clip(clip);

        player.seek_frame(15);
        assert!((player.current_time() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_next_frame() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.frame_rate = 30.0;
        let mut player = ClipPlayer::with_clip(clip);

        player.seek_frame(10);
        player.next_frame();
        assert_eq!(player.current_frame(), 11);
    }

    #[test]
    fn test_prev_frame() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.frame_rate = 30.0;
        let mut player = ClipPlayer::with_clip(clip);

        player.seek_frame(10);
        player.prev_frame();
        assert_eq!(player.current_frame(), 9);
    }

    #[test]
    fn test_prev_frame_at_start() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.frame_rate = 30.0;
        let mut player = ClipPlayer::with_clip(clip);

        player.prev_frame();
        assert_eq!(player.current_frame(), 0);
    }

    // =========================================================================
    // State Query Tests
    // =========================================================================

    #[test]
    fn test_is_playing() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        assert!(!player.is_playing());
        player.play();
        assert!(player.is_playing());
        player.pause();
        assert!(!player.is_playing());
    }

    #[test]
    fn test_is_paused() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        assert!(!player.is_paused());
        player.play();
        assert!(!player.is_paused());
        player.pause();
        assert!(player.is_paused());
    }

    #[test]
    fn test_is_stopped() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        assert!(player.is_stopped());
        player.play();
        assert!(!player.is_stopped());
        player.stop();
        assert!(player.is_stopped());
    }

    #[test]
    fn test_is_completed() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();

        assert!(!player.is_completed());
        player.update(1.5);
        assert!(player.is_completed());
    }

    // =========================================================================
    // Play from Paused State Tests
    // =========================================================================

    #[test]
    fn test_play_from_paused_continues() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.3);
        player.pause();

        let time_before = player.current_time();
        player.play(); // Should resume, not restart

        assert_eq!(player.current_time(), time_before);
        assert_eq!(player.state(), PlaybackState::Playing);
    }

    // =========================================================================
    // Complex Playback Scenarios
    // =========================================================================

    #[test]
    fn test_ping_pong_with_rate_change() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::PingPong);
        player.play();

        player.update(1.5); // At 0.5 going backward
        assert_eq!(player.direction, -1.0);

        // Change rate to reverse (which means forward in ping-pong reverse direction)
        player.set_play_rate(-1.0);
        player.update(0.2);

        // Should now move forward (reverse of reverse)
        assert!(player.current_time() > 0.5);
    }

    #[test]
    fn test_loop_mode_changed_mid_playback() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.set_loop_mode(LoopMode::Loop);
        player.play();

        player.update(0.8);

        // Change to Once mode
        player.set_loop_mode(LoopMode::Once);
        player.update(0.5);

        // Should stop at duration
        assert_eq!(player.current_time(), 1.0);
        assert!(player.is_completed());
    }

    #[test]
    fn test_rapid_state_transitions() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));

        player.play();
        player.pause();
        player.resume();
        player.stop();
        player.play();
        player.toggle_pause();
        player.toggle_pause();

        assert_eq!(player.state(), PlaybackState::Playing);
    }

    // =========================================================================
    // Clone and Debug Tests
    // =========================================================================

    #[test]
    fn test_clip_player_clone() {
        let mut player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        player.play();
        player.update(0.5);

        let cloned = player.clone();

        assert_eq!(cloned.current_time(), player.current_time());
        assert_eq!(cloned.state(), player.state());
    }

    #[test]
    fn test_clip_player_debug() {
        let player = ClipPlayer::with_clip(AnimationClip::new("test", 1.0));
        let debug = format!("{:?}", player);
        assert!(debug.contains("ClipPlayer"));
    }

    // =========================================================================
    // Seek with Events Tests
    // =========================================================================

    #[test]
    fn test_seek_fires_events() {
        let mut player = ClipPlayer::with_clip(create_clip_with_events());

        // Seek from 0 to 0.6 should fire events at 0, 0.25, 0.5
        player.seek(0.6);
        let events = player.drain_events();

        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_seek_backward_fires_events() {
        let mut player = ClipPlayer::with_clip(create_clip_with_events());
        player.set_time(0.8);

        // Seek backward from 0.8 to 0.2
        player.seek(0.2);
        let events = player.drain_events();

        // Should fire events in the range [0.2, 0.8)
        let event_names: Vec<_> = events.iter().map(|e| e.name.as_str()).collect();
        assert!(event_names.contains(&"quarter"));
        assert!(event_names.contains(&"half"));
        assert!(event_names.contains(&"three_quarter"));
    }
}
