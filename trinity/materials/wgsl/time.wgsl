// Time Uniforms and Animation Functions for TRINITY Material System
// T-MAT-5.5: Material Animation System
//
// This module provides:
//   - TimeUniforms struct with elapsed time, delta time, and frame count
//   - Procedural animation functions (sin wave, cos wave, sawtooth, pulse)
//   - UV animation functions (scrolling, rotation, oscillation)
//   - Color animation functions (pulse, flicker, cycle)
//   - Emission animation functions (flicker, pulse, breathe)
//
// Usage in shaders:
//   let t = time_uniforms.elapsed_seconds;
//   let animated_uv = animate_uv_scroll(uv, vec2<f32>(0.5, 0.0), t);
//   let pulsing_color = animate_color_pulse(base_color, 2.0, 0.3, t);

// Mathematical constants
const TAU: f32 = 6.28318530718;  // 2 * PI
const ANIM_PI: f32 = 3.14159265359;

// =============================================================================
// TIME UNIFORMS STRUCT
// =============================================================================

/// Time-related uniform values updated per frame.
/// Typically bound to a uniform buffer at binding group 0.
struct TimeUniforms {
    /// Total elapsed time in seconds since animation start
    elapsed_seconds: f32,
    /// Time delta since last frame in seconds
    delta_time: f32,
    /// Frame counter (wraps at u32::MAX)
    frame_count: u32,
    /// Padding for 16-byte alignment
    _padding: u32,
}

// =============================================================================
// BASIC WAVE FUNCTIONS
// =============================================================================

/// Sine wave oscillation.
/// Returns value in range [-1, 1].
///
/// @param t: Time in seconds
/// @param frequency: Oscillation frequency in Hz
/// @param phase: Phase offset in radians
/// @returns: Sine wave value in [-1, 1]
fn sin_wave(t: f32, frequency: f32, phase: f32) -> f32 {
    return sin(t * frequency * TAU + phase);
}

/// Cosine wave oscillation.
/// Returns value in range [-1, 1].
///
/// @param t: Time in seconds
/// @param frequency: Oscillation frequency in Hz
/// @param phase: Phase offset in radians
/// @returns: Cosine wave value in [-1, 1]
fn cos_wave(t: f32, frequency: f32, phase: f32) -> f32 {
    return cos(t * frequency * TAU + phase);
}

/// Normalized sine wave.
/// Returns value in range [0, 1].
///
/// @param t: Time in seconds
/// @param frequency: Oscillation frequency in Hz
/// @param phase: Phase offset in radians
/// @returns: Normalized sine wave value in [0, 1]
fn sin_wave_01(t: f32, frequency: f32, phase: f32) -> f32 {
    return sin_wave(t, frequency, phase) * 0.5 + 0.5;
}

/// Normalized cosine wave.
/// Returns value in range [0, 1].
///
/// @param t: Time in seconds
/// @param frequency: Oscillation frequency in Hz
/// @param phase: Phase offset in radians
/// @returns: Normalized cosine wave value in [0, 1]
fn cos_wave_01(t: f32, frequency: f32, phase: f32) -> f32 {
    return cos_wave(t, frequency, phase) * 0.5 + 0.5;
}

/// Sawtooth wave (linear ramp).
/// Returns value in range [0, 1] that increases linearly then resets.
///
/// @param t: Time in seconds
/// @param period: Period of the wave in seconds
/// @returns: Sawtooth value in [0, 1]
fn sawtooth(t: f32, period: f32) -> f32 {
    return fract(t / period);
}

/// Reverse sawtooth wave.
/// Returns value in range [0, 1] that decreases linearly then resets.
///
/// @param t: Time in seconds
/// @param period: Period of the wave in seconds
/// @returns: Reverse sawtooth value in [0, 1]
fn sawtooth_reverse(t: f32, period: f32) -> f32 {
    return 1.0 - fract(t / period);
}

/// Triangle wave.
/// Returns value in range [0, 1] that goes up then down.
///
/// @param t: Time in seconds
/// @param period: Period of the wave in seconds
/// @returns: Triangle wave value in [0, 1]
fn triangle_wave(t: f32, period: f32) -> f32 {
    let saw = sawtooth(t, period);
    return 1.0 - abs(saw * 2.0 - 1.0);
}

/// Pulse/square wave.
/// Returns 0 or 1 with controllable duty cycle.
///
/// @param t: Time in seconds
/// @param period: Period of the wave in seconds
/// @param duty: Duty cycle in [0, 1], fraction of period that output is 1
/// @returns: 0.0 or 1.0
fn pulse(t: f32, period: f32, duty: f32) -> f32 {
    let phase = fract(t / period);
    return select(0.0, 1.0, phase < duty);
}

/// Smooth pulse using sine interpolation.
/// Returns smooth transition between 0 and 1.
///
/// @param t: Time in seconds
/// @param period: Period of the wave in seconds
/// @param attack: Attack time as fraction of period [0, 0.5]
/// @param release: Release time as fraction of period [0, 0.5]
/// @returns: Smooth pulse value in [0, 1]
fn smooth_pulse(t: f32, period: f32, attack: f32, release: f32) -> f32 {
    let phase = fract(t / period);
    let hold_start = attack;
    let hold_end = 1.0 - release;

    if phase < attack {
        // Attack phase: smooth ramp up
        return smoothstep(0.0, attack, phase);
    } else if phase < hold_end {
        // Hold phase: full on
        return 1.0;
    } else {
        // Release phase: smooth ramp down
        return 1.0 - smoothstep(hold_end, 1.0, phase);
    }
}

// =============================================================================
// NOISE-BASED ANIMATION
// =============================================================================

/// Simple hash for procedural noise.
fn anim_hash(p: f32) -> f32 {
    var x = fract(p * 0.1031);
    x = x * (x + 33.33);
    return fract((x + x) * x);
}

/// 2D hash for procedural noise.
fn anim_hash2(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

/// Noise-based animation.
/// Returns pseudo-random value that changes smoothly over time.
///
/// @param t: Time in seconds
/// @param frequency: Base frequency for noise
/// @param octaves: Number of noise octaves (1-4)
/// @returns: Noise value in [0, 1]
fn noise_anim(t: f32, frequency: f32, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var total_amplitude = 0.0;
    var freq = frequency;

    for (var i = 0; i < octaves && i < 4; i = i + 1) {
        let floor_t = floor(t * freq);
        let fract_t = fract(t * freq);
        // Smooth interpolation using smoothstep
        let smooth_t = fract_t * fract_t * (3.0 - 2.0 * fract_t);
        let a = anim_hash(floor_t);
        let b = anim_hash(floor_t + 1.0);
        value = value + mix(a, b, smooth_t) * amplitude;
        total_amplitude = total_amplitude + amplitude;
        amplitude = amplitude * 0.5;
        freq = freq * 2.0;
    }

    return value / total_amplitude;
}

/// Flicker animation using high-frequency noise.
/// Good for fire, electricity, or random glitching.
///
/// @param t: Time in seconds
/// @param intensity: Flicker intensity in [0, 1]
/// @param speed: Flicker speed multiplier
/// @returns: Flicker value in [0, 1]
fn flicker(t: f32, intensity: f32, speed: f32) -> f32 {
    let noise = noise_anim(t, speed * 10.0, 2);
    return mix(1.0, noise, intensity);
}

// =============================================================================
// UV ANIMATION FUNCTIONS
// =============================================================================

/// Animate UV coordinates with linear scrolling.
///
/// @param uv: Original UV coordinates
/// @param speed: Scroll speed (units per second) for each axis
/// @param t: Time in seconds
/// @returns: Animated UV coordinates
fn animate_uv_scroll(uv: vec2<f32>, speed: vec2<f32>, t: f32) -> vec2<f32> {
    return fract(uv + speed * t);
}

/// Animate UV coordinates with scrolling (no wrap).
///
/// @param uv: Original UV coordinates
/// @param speed: Scroll speed (units per second) for each axis
/// @param t: Time in seconds
/// @returns: Animated UV coordinates (may exceed [0,1])
fn animate_uv_scroll_no_wrap(uv: vec2<f32>, speed: vec2<f32>, t: f32) -> vec2<f32> {
    return uv + speed * t;
}

/// Animate UV coordinates with rotation around center.
///
/// @param uv: Original UV coordinates
/// @param speed: Rotation speed in radians per second
/// @param t: Time in seconds
/// @returns: Rotated UV coordinates
fn animate_uv_rotate(uv: vec2<f32>, speed: f32, t: f32) -> vec2<f32> {
    let centered = uv - 0.5;
    let angle = speed * t;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let rotated = vec2<f32>(
        centered.x * cos_a - centered.y * sin_a,
        centered.x * sin_a + centered.y * cos_a
    );
    return rotated + 0.5;
}

/// Animate UV coordinates with rotation around arbitrary pivot.
///
/// @param uv: Original UV coordinates
/// @param pivot: Rotation center in UV space
/// @param speed: Rotation speed in radians per second
/// @param t: Time in seconds
/// @returns: Rotated UV coordinates
fn animate_uv_rotate_pivot(uv: vec2<f32>, pivot: vec2<f32>, speed: f32, t: f32) -> vec2<f32> {
    let centered = uv - pivot;
    let angle = speed * t;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let rotated = vec2<f32>(
        centered.x * cos_a - centered.y * sin_a,
        centered.x * sin_a + centered.y * cos_a
    );
    return rotated + pivot;
}

/// Animate UV coordinates with oscillating offset (ping-pong).
///
/// @param uv: Original UV coordinates
/// @param amplitude: Maximum offset amplitude for each axis
/// @param frequency: Oscillation frequency in Hz
/// @param t: Time in seconds
/// @returns: Oscillating UV coordinates
fn animate_uv_oscillate(uv: vec2<f32>, amplitude: vec2<f32>, frequency: f32, t: f32) -> vec2<f32> {
    let offset = amplitude * sin(t * frequency * TAU);
    return uv + offset;
}

/// Animate UV coordinates with wave distortion.
///
/// @param uv: Original UV coordinates
/// @param amplitude: Wave amplitude
/// @param frequency: Wave frequency
/// @param speed: Wave propagation speed
/// @param t: Time in seconds
/// @returns: Wave-distorted UV coordinates
fn animate_uv_wave(uv: vec2<f32>, amplitude: f32, frequency: f32, speed: f32, t: f32) -> vec2<f32> {
    let wave_x = sin(uv.y * frequency * TAU + t * speed) * amplitude;
    let wave_y = sin(uv.x * frequency * TAU + t * speed) * amplitude;
    return uv + vec2<f32>(wave_x, wave_y);
}

/// Animate UV with scale pulsing (zoom effect).
///
/// @param uv: Original UV coordinates
/// @param scale_range: Scale range (min, max)
/// @param frequency: Pulse frequency in Hz
/// @param t: Time in seconds
/// @returns: Scaled UV coordinates
fn animate_uv_scale(uv: vec2<f32>, scale_range: vec2<f32>, frequency: f32, t: f32) -> vec2<f32> {
    let scale = mix(scale_range.x, scale_range.y, sin_wave_01(t, frequency, 0.0));
    let centered = uv - 0.5;
    return centered / scale + 0.5;
}

// =============================================================================
// COLOR ANIMATION FUNCTIONS
// =============================================================================

/// Animate color with intensity pulse.
///
/// @param color: Base color (RGB)
/// @param frequency: Pulse frequency in Hz
/// @param intensity: Pulse intensity (0 = no pulse, 1 = full black to white)
/// @param t: Time in seconds
/// @returns: Pulsing color
fn animate_color_pulse(color: vec3<f32>, frequency: f32, intensity: f32, t: f32) -> vec3<f32> {
    let factor = sin_wave_01(t, frequency, 0.0) * intensity + (1.0 - intensity);
    return color * factor;
}

/// Animate color with additive pulse.
///
/// @param color: Base color (RGB)
/// @param pulse_color: Color to pulse toward
/// @param frequency: Pulse frequency in Hz
/// @param t: Time in seconds
/// @returns: Pulsing color
fn animate_color_pulse_add(color: vec3<f32>, pulse_color: vec3<f32>, frequency: f32, t: f32) -> vec3<f32> {
    let factor = sin_wave_01(t, frequency, 0.0);
    return mix(color, pulse_color, factor);
}

/// Animate color hue cycling.
///
/// @param saturation: Color saturation [0, 1]
/// @param value: Color value/brightness [0, 1]
/// @param speed: Hue cycling speed (cycles per second)
/// @param t: Time in seconds
/// @returns: Color with animated hue
fn animate_color_hue_cycle(saturation: f32, value: f32, speed: f32, t: f32) -> vec3<f32> {
    let hue = fract(t * speed);
    // HSV to RGB conversion
    let c = value * saturation;
    let x = c * (1.0 - abs(fract(hue * 6.0) * 2.0 - 1.0));
    let m = value - c;

    var rgb: vec3<f32>;
    let h = hue * 6.0;
    if h < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if h < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if h < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if h < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if h < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + m;
}

/// Animate color with random flicker (fire/electricity effect).
///
/// @param color: Base color (RGB)
/// @param intensity: Flicker intensity [0, 1]
/// @param speed: Flicker speed multiplier
/// @param t: Time in seconds
/// @returns: Flickering color
fn animate_color_flicker(color: vec3<f32>, intensity: f32, speed: f32, t: f32) -> vec3<f32> {
    let flicker_val = flicker(t, intensity, speed);
    return color * flicker_val;
}

/// Animate color temperature shift (warm to cool).
///
/// @param base_temp: Base color temperature (0 = cool/blue, 1 = warm/orange)
/// @param amplitude: Temperature shift amplitude
/// @param frequency: Shift frequency in Hz
/// @param t: Time in seconds
/// @returns: Temperature-shifted color
fn animate_color_temperature(base_temp: f32, amplitude: f32, frequency: f32, t: f32) -> vec3<f32> {
    let temp = clamp(base_temp + sin_wave(t, frequency, 0.0) * amplitude, 0.0, 1.0);
    // Simple temperature to RGB (approximate blackbody)
    let warm = vec3<f32>(1.0, 0.7, 0.4);
    let cool = vec3<f32>(0.4, 0.6, 1.0);
    return mix(cool, warm, temp);
}

// =============================================================================
// EMISSION ANIMATION FUNCTIONS
// =============================================================================

/// Animate emission with flickering effect (candle/fire).
///
/// @param base_emission: Base emission color
/// @param frequency: Base flicker frequency
/// @param intensity: Flicker intensity [0, 1]
/// @param t: Time in seconds
/// @returns: Flickering emission
fn animate_emission_flicker(base_emission: vec3<f32>, frequency: f32, intensity: f32, t: f32) -> vec3<f32> {
    // Multi-frequency flicker for more natural look
    let f1 = sin_wave_01(t, frequency, 0.0);
    let f2 = sin_wave_01(t, frequency * 1.7, 0.5);
    let f3 = noise_anim(t, frequency * 3.0, 2);

    let combined = (f1 * 0.4 + f2 * 0.3 + f3 * 0.3);
    let factor = mix(1.0, combined, intensity);

    return base_emission * factor;
}

/// Animate emission with pulsing effect (heartbeat/power core).
///
/// @param base_emission: Base emission color
/// @param frequency: Pulse frequency in Hz
/// @param min_intensity: Minimum emission intensity [0, 1]
/// @param max_intensity: Maximum emission intensity [0, inf]
/// @param t: Time in seconds
/// @returns: Pulsing emission
fn animate_emission_pulse(
    base_emission: vec3<f32>,
    frequency: f32,
    min_intensity: f32,
    max_intensity: f32,
    t: f32
) -> vec3<f32> {
    let factor = mix(min_intensity, max_intensity, sin_wave_01(t, frequency, 0.0));
    return base_emission * factor;
}

/// Animate emission with breathing effect (smooth, organic pulse).
///
/// @param base_emission: Base emission color
/// @param period: Breath cycle period in seconds
/// @param min_intensity: Minimum emission intensity
/// @param max_intensity: Maximum emission intensity
/// @param t: Time in seconds
/// @returns: Breathing emission
fn animate_emission_breathe(
    base_emission: vec3<f32>,
    period: f32,
    min_intensity: f32,
    max_intensity: f32,
    t: f32
) -> vec3<f32> {
    // Asymmetric breathing curve: slow in, fast out
    let phase = fract(t / period);
    var factor: f32;
    if phase < 0.4 {
        // Inhale (40% of cycle, slow)
        factor = smoothstep(0.0, 0.4, phase);
    } else {
        // Exhale (60% of cycle, faster)
        factor = 1.0 - smoothstep(0.4, 1.0, phase);
    }
    return base_emission * mix(min_intensity, max_intensity, factor);
}

/// Animate emission with strobe effect (rapid on/off).
///
/// @param base_emission: Base emission color
/// @param frequency: Strobe frequency in Hz
/// @param duty: Duty cycle [0, 1], fraction of time light is on
/// @param t: Time in seconds
/// @returns: Strobing emission
fn animate_emission_strobe(base_emission: vec3<f32>, frequency: f32, duty: f32, t: f32) -> vec3<f32> {
    let on = pulse(t, 1.0 / frequency, duty);
    return base_emission * on;
}

/// Animate emission with warning effect (alternating intensity).
///
/// @param base_emission: Base emission color
/// @param frequency: Warning frequency in Hz
/// @param t: Time in seconds
/// @returns: Warning emission
fn animate_emission_warning(base_emission: vec3<f32>, frequency: f32, t: f32) -> vec3<f32> {
    // Double-pulse pattern for warning
    let period = 1.0 / frequency;
    let phase = fract(t / period);

    var factor: f32;
    if phase < 0.15 {
        factor = smoothstep(0.0, 0.1, phase);
    } else if phase < 0.25 {
        factor = 1.0 - smoothstep(0.15, 0.25, phase);
    } else if phase < 0.35 {
        factor = smoothstep(0.25, 0.35, phase);
    } else if phase < 0.5 {
        factor = 1.0 - smoothstep(0.35, 0.5, phase);
    } else {
        factor = 0.0;
    }

    return base_emission * factor;
}

// =============================================================================
// COMPOSITE ANIMATION HELPERS
// =============================================================================

/// Combine multiple animation factors with blending.
///
/// @param a: First animation factor
/// @param b: Second animation factor
/// @param blend: Blend mode (0 = multiply, 1 = add, 2 = max, 3 = min)
/// @returns: Combined factor
fn combine_animation(a: f32, b: f32, blend: i32) -> f32 {
    switch blend {
        case 0: { return a * b; }
        case 1: { return saturate(a + b); }
        case 2: { return max(a, b); }
        case 3: { return min(a, b); }
        default: { return a * b; }
    }
}

/// Ease-in-out interpolation.
///
/// @param t: Linear time [0, 1]
/// @returns: Eased value [0, 1]
fn ease_in_out(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

/// Bounce interpolation.
///
/// @param t: Linear time [0, 1]
/// @returns: Bouncing value [0, 1+]
fn ease_bounce(t: f32) -> f32 {
    let t2 = t * t;
    return t2 * t * (t2 * 6.0 - 15.0 * t + 10.0) + sin(t * ANIM_PI * 4.0) * (1.0 - t) * 0.1;
}

/// Elastic interpolation.
///
/// @param t: Linear time [0, 1]
/// @param amplitude: Overshoot amplitude
/// @param period: Oscillation period
/// @returns: Elastic value
fn ease_elastic(t: f32, amplitude: f32, period: f32) -> f32 {
    if t <= 0.0 { return 0.0; }
    if t >= 1.0 { return 1.0; }
    let s = period / TAU * asin(1.0 / amplitude);
    return amplitude * pow(2.0, -10.0 * t) * sin((t - s) * TAU / period) + 1.0;
}
