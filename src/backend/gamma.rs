//! Color science implementation for accurate color temperature calculations.
//!
//! This module provides sophisticated colorimetric calculations for converting
//! color temperatures to gamma table adjustments.
//!
//! ## Attribution
//!
//! This is an adaptation of wlsunset's color temperature calculation algorithm.
//! Original C implementation: <https://git.sr.ht/~kennylevinsen/wlsunset>
//!
//! wlsunset uses proper colorimetric calculations with CIE XYZ color space,
//! planckian locus, and illuminant D curves to produce accurate color temperatures.
//! This approach is much more accurate than simple RGB approximations, so we've
//! adopted it with some minor adaptations to extend the bottom end of the range.
//!
//! ## Implementation Details
//!
//! The module performs several color space transformations:
//! 1. **Planckian Locus**: Calculates the theoretical color of a black body at a given temperature
//! 2. **CIE XYZ Color Space**: Uses the standard colorimetric system for device-independent colors
//! 3. **sRGB Conversion**: Transforms to the standard RGB color space used by displays
//! 4. **Gamma Correction**: Applies proper gamma curves for display linearization
//!
//! ## Temperature Range
//!
//! Sunsetr supports color temperatures from 1000K to 20000K:
//! - **1000K-1666K:** Tanner Helland empirical approximation (deep reds)
//! - **1667K-2500K:** Planckian locus (warm incandescent-like colors)
//! - **2500K-4000K:** Smooth transition between planckian and illuminant D
//! - **4000K-20000K:** Illuminant D (daylight simulation)
//!
//! The hybrid approach ensures continuous coverage across the full range
//! while maintaining scientific accuracy where CIE colorimetry is valid.

use anyhow::Result;

/// RGB color representation (0.0 to 1.0 range)
#[derive(Debug, Clone, Copy)]
struct Rgb {
    r: f64,
    g: f64,
    b: f64,
}

/// XY chromaticity representation
#[derive(Debug, Clone, Copy)]
struct Xy {
    x: f64,
    y: f64,
}

/// XYZ color space representation
#[derive(Debug, Clone, Copy)]
struct Xyz {
    x: f64,
    y: f64,
    z: f64,
}

/// Clamp value to 0.0-1.0 range
// fn clamp(value: f64) -> f64 {
//     value.clamp(0.0, 1.0)
// }

fn oetf_gamma_22(value: f64) -> f64 {
    value.powf(1.0 / 2.2)
}

fn oetf_gamma_26(value: f64) -> f64 {
    value.powf(1.0 / 2.6)
}

fn oetf_srgb(value: f64) -> f64 {
    if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn oetf_rec2020(value: f64) -> f64 {
    if value < 0.018 {
        4.5 * value
    } else {
        1.099 * value.powf(0.45) - 0.099
    }
}

/// Apply opto-electronic transfer function
/// https://en.wikipedia.org/wiki/Transfer_functions_in_imaging
fn apply_oetf(rgb: &mut Rgb, oetf: fn(f64) -> f64) {
    rgb.r = oetf(rgb.r);
    rgb.g = oetf(rgb.g);
    rgb.b = oetf(rgb.b);
}

/// https://wiki.hypr.land/Configuring/Monitors/#color-management-presets
/// Reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
const MATRIX_XYZ_TO_REC709: [f64; 9] = [
    3.2404542, -1.5371385, -0.4985314,
    -0.9692660, 1.8760108, 0.0415560,
    0.0556434, -0.2040259, 1.0572252,
];

const MATRIX_XYZ_TO_REC2020: [f64; 9] = [
    1.4628067, -0.1840623, -0.2743606,
    -0.5217933, 1.4472381, 0.0677227,
    0.0349342, -0.0968930, 1.2884099,
];

const MATRIX_XYZ_TO_P3: [f64; 9] = [
    2.7253940, -1.01800301, -0.4401631,
    -0.7951680, 1.6897321, 0.0226472,
    0.0412419, -0.0876390, 1.1009294,
];

const MATRIX_XYZ_TO_ADOBE: [f64; 9] = [
    2.0413690, -0.5649464, -0.3446944,
    -0.9692660, 1.8760108, 0.0415560,
    0.0134474, -0.1183897, 1.0154096,
];

fn linear_transformation(xyz: &Xyz, m: [f64; 9]) -> Rgb {
    Rgb {
        r: m[0] * xyz.x + m[1] * xyz.y + m[2] * xyz.z,
        g: m[3] * xyz.x + m[4] * xyz.y + m[5] * xyz.z,
        b: m[6] * xyz.x + m[7] * xyz.y + m[8] * xyz.z,
    }
}

/// Convert XYZ color space to RGB using standard transformation matrix
/// Reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
fn xyz_to_rbg(xyz: &Xyz, gamut: &Gamut) -> Rgb {
    let matrix = match gamut {
        Gamut::Rec709 => MATRIX_XYZ_TO_REC709,
        Gamut::Rec2020 => MATRIX_XYZ_TO_REC2020,
        Gamut::P3 => MATRIX_XYZ_TO_P3,
        Gamut::Adobe => MATRIX_XYZ_TO_ADOBE,
    };
    linear_transformation(xyz, matrix)
}

/// Reference: https://iquilezles.org/articles/smin/
fn smooth_min(a: f64, b: f64, falloff: f64) -> f64 {
    let k = 6.0 * falloff;
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - k * h * h * h / 6.0
}

fn smooth_max(a: f64, b: f64, falloff: f64) -> f64 {
    -smooth_min(-a, -b, falloff)
}

fn gamut_compression(rgb: &mut Rgb, falloff: f64) {
    rgb.r = smooth_max(rgb.r, 0.0, falloff).min(1.0);
    rgb.g = smooth_max(rgb.g, 0.0, falloff).min(1.0);
    rgb.b = smooth_max(rgb.b, 0.0, falloff).min(1.0);
}

/// Normalize RGB so the maximum component is 1.0
fn rgb_normalize(rgb: &mut Rgb) {
    let max_component = rgb.r.max(rgb.g.max(rgb.b));
    if max_component > 0.0 {
        rgb.r /= max_component;
        rgb.g /= max_component;
        rgb.b /= max_component;
    }
}

/// Multiply RGB component-wise
fn rgb_brightness(rgb: &mut Rgb, brightness: f64) {
    rgb.r *= brightness;
    rgb.g *= brightness;
    rgb.b *= brightness;
}

/// Laurent polynomial function going from power of -3 to 3
fn temp_to_chroma_fit_curve(x: f64, c: [f64; 7]) -> f64 {
    (0..6).rev().fold(c[6], |total, i| total * x + c[i]) / (x * x * x)
}

/// Calculate Planckian locus chromaticity coordinates
///
/// Planckian locus (black body locus) describes the color of a black body
/// at a certain temperature directly at its source. This is how we expect
/// dim, warm light sources (like incandescent bulbs) to look.
/// Valid range: 1000 to 20000K
///
/// Reference: https://en.wikipedia.org/wiki/Planckian_locus#Approximation
fn temperature_to_chroma(temp: f64) -> Xy {
    const COEFFS_X: [f64; 7] = [
        4.60243e+08, -1.34958e+06, 1.49958e+03, 2.20742e-02, 1.86755e-05, -7.48912e-10, 1.12218e-14
    ];
    const COEFFS_Y: [f64; 7] = [
        8.19188e-02, -1.32154e+00, 8.63682e+00, -2.95048e+01, 5.67579e+01, -5.42917e+01, 1.98083e+01
    ];

    let chroma_x = temp_to_chroma_fit_curve(temp, COEFFS_X);
    let chroma_y = temp_to_chroma_fit_curve(chroma_x, COEFFS_Y);

    Xy {x: chroma_x, y: chroma_y}
}

/// https://en.wikipedia.org/wiki/Smoothstep
fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    3.0 * x.powi(2) - 2.0 * x.powi(3)
}

/// Convert temperature from Kelvins to Mireds
fn k_to_mired(kelvin: u32) -> u32 {
    1000000 / kelvin
}

enum ColorSpace {
    SRGB,
    Wide,
    DP3,
    DciP3,
    Adobe,
}

enum Gamut {
    Rec709,
    Rec2020,
    P3,
    Adobe,
}

enum Illuminant {
    D63,
    D65,
}

fn get_oetf(color_space: &ColorSpace) -> fn(f64) -> f64 {
    match color_space {
        ColorSpace::SRGB => oetf_srgb,
        ColorSpace::Wide => oetf_rec2020,
        ColorSpace::DciP3 => oetf_gamma_26,
        ColorSpace::DP3 => oetf_srgb,
        ColorSpace::Adobe => oetf_gamma_22,
    }
}

fn get_illuminant(color_space: &ColorSpace) -> Illuminant {
    match color_space {
        ColorSpace::SRGB => Illuminant::D65,
        ColorSpace::Wide => Illuminant::D65,
        ColorSpace::DciP3 => Illuminant::D63,
        ColorSpace::DP3 => Illuminant::D65,
        ColorSpace::Adobe => Illuminant::D65,
    }
}

fn get_gamut(color_space: &ColorSpace) -> Gamut {
    match color_space {
        ColorSpace::SRGB => Gamut::Rec709,
        ColorSpace::Wide => Gamut::Rec2020,
        ColorSpace::DciP3 => Gamut::P3,
        ColorSpace::DP3 => Gamut::P3,
        ColorSpace::Adobe => Gamut::Adobe,
    }
}

fn get_wp_cct(illuminant: &Illuminant) -> u32 {
    match illuminant {
        Illuminant::D63 => 6300,
        Illuminant::D65 => 6500,
    }
}

fn get_wp_chroma(illuminant: &Illuminant) -> Xy {
    match illuminant {
        Illuminant::D63 => Xy { x: 0.3140, y: 0.3510 },
        Illuminant::D65 => Xy { x: 0.3127, y: 0.3290 },
    }
}

/// Calculate white point RGB values for a given color temperature
///
/// This is an adaptation of wlsunset's calc_whitepoint function.
/// It uses a combination of planckian locus (for warm temperatures),
/// illuminant D (for cool temperatures), and Tanner Helland's algorithm
/// (for deep reds).
///
/// The algorithm smoothly transitions between planckian locus and illuminant D
/// in the 2500K-4000K range to provide subjectively pleasant colors and uses
/// Tanner Helland's method to extend the range down from 1667K to 1000K.
pub fn temperature_to_rgb(temp: u32, brightness: f64) -> (f32, f32, f32) {
    // TODO: pass color space
    let color_space = ColorSpace::SRGB;
    let illuminant = get_illuminant(&color_space);

    let expected_wp_temp = get_wp_cct(&illuminant);
    let expected_wp_chroma = get_wp_chroma(&illuminant);

    // if temp == expected_wp_temp {
    //     return (1.0, 1.0, 1.0);
    // }

    let chroma_at_wp = temperature_to_chroma(expected_wp_temp as f64);
    let chroma_at_temp = temperature_to_chroma(temp as f64);

    // Offset the locus so it intersects the monitor's whitepoint exactly.
    // That way, when a user sets 5000K on a D50 monitor, they'll get "true white"
    // but different temperatures will blend into the Planckian locus.
    const FALLOFF: f64 = 150.0;
    let offset_weight = smoothstep(
        1.0 - k_to_mired(temp).abs_diff(k_to_mired(expected_wp_temp)) as f64 / FALLOFF
    );

    // Construct chromaticity coordinates
    let wp = Xy {
        x: chroma_at_temp.x + offset_weight * (expected_wp_chroma.x - chroma_at_wp.x),
        y: chroma_at_temp.y + offset_weight * (expected_wp_chroma.y - chroma_at_wp.y),
    };

    // Convert chromaticity coordinates to XYZ
    let wp_z = 1.0 - wp.x - wp.y;
    let xyz = Xyz {
        x: wp.x,
        y: wp.y,
        z: wp_z,
    };

    // Convert XYZ to RGB
    let gamut = get_gamut(&color_space);
    let mut rgb = xyz_to_rbg(&xyz, &gamut);

    // Normalize and apply brightness
    rgb_normalize(&mut rgb);
    rgb_brightness(&mut rgb, brightness.powi(3));

    // Compress gamut for softer falloff at the gamut boundary
    gamut_compression(&mut rgb, 0.005);

    // Apply Opto-Electronic Transfer Function to compensate for compositor limitations
    let oetf = get_oetf(&color_space);
    apply_oetf(&mut rgb, oetf);

    // Return as f32 for compatibility with existing code
    (rgb.r as f32, rgb.g as f32, rgb.b as f32)
}

// # End of wlsunset Color Science Implementation

/// Get RGB factors for a given color temperature as a formatted tuple.
/// This is a convenience function for debug logging.
///
/// # Arguments
/// * `temperature` - Color temperature in Kelvin (1000-25000)
///
/// # Returns
/// Tuple of (red_factor, green_factor, blue_factor) rounded to 3 decimal places
pub fn get_rgb_factors(temperature: u32) -> (f32, f32, f32) {
    let (r, g, b) = temperature_to_rgb(temperature, 1.0);
    // Round to 3 decimal places for cleaner logging
    (
        (r * 1000.0).round() / 1000.0,
        (g * 1000.0).round() / 1000.0,
        (b * 1000.0).round() / 1000.0,
    )
}

/// Generate gamma table for a specific color channel using wlsunset's approach.
///
/// Creates a gamma lookup table (LUT) that maps input values to output values
/// using a power function gamma curve.
///
/// # Arguments
/// * `size` - Size of the gamma table (typically 256 or 1024)
/// * `color_factor` - Color temperature adjustment factor (0.0-1.0)
/// * `gamma` - Gamma curve value (typically 1.0 for linear, 0.9 for 90% brightness)
///
/// # Returns
/// Vector of 16-bit gamma values for this color channel
pub fn generate_gamma_table(size: usize, color_factor: f64, gamma: f64) -> Vec<u16> {
    let mut table = Vec::with_capacity(size);

    for i in 0..size {
        // Calculate normalized input value (0.0 to 1.0)
        let val = i as f64 / (size - 1) as f64;

        // Apply color temperature factor and gamma curve using power function
        let output = (val * color_factor * 65535.0).clamp(0.0, 65535.0);

        table.push(output as u16);
    }

    table
}

/// Create complete gamma tables for RGB channels.
///
/// Generates the full set of gamma lookup tables needed for the
/// wlr-gamma-control-unstable-v1 protocol.
///
/// # Arguments
/// * `size` - Size of each gamma table (reported by compositor)
/// * `temperature` - Color temperature in Kelvin
/// * `gamma_percent` - Gamma adjustment as percentage (90% = 0.9, 100% = 1.0, 200% = 2.0)
/// * `debug_enabled` - Whether to output debug information
///
/// # Returns
/// Byte vector containing concatenated R, G, B gamma tables
pub fn create_gamma_tables(
    size: usize,
    temperature: u32,
    gamma_percent: f32,
    debug_enabled: bool,
) -> Result<Vec<u8>> {
    // Convert temperature to RGB factors
    let (red_factor, green_factor, blue_factor) = temperature_to_rgb(temperature, gamma_percent as f64 / 100.0);

    // Generate individual channel tables using power function gamma curves
    let red_table = generate_gamma_table(size, red_factor as f64, gamma_percent as f64);
    let green_table = generate_gamma_table(size, green_factor as f64, gamma_percent as f64);
    let blue_table = generate_gamma_table(size, blue_factor as f64, gamma_percent as f64);

    // Log some sample values for debugging
    if debug_enabled {
        let sample_indices = [0, 10, 128, 255];
        let r_samples: Vec<u16> = sample_indices.iter().map(|&idx| red_table[idx]).collect();
        let g_samples: Vec<u16> = sample_indices.iter().map(|&idx| green_table[idx]).collect();
        let b_samples: Vec<u16> = sample_indices.iter().map(|&idx| blue_table[idx]).collect();

        log_decorated!("Sample gamma values:");
        log_indented!("R: {:?}", r_samples);
        log_indented!("G: {:?}", g_samples);
        log_indented!("B: {:?}", b_samples);
    }

    // Convert to bytes (little-endian 16-bit values)
    // Using the documented wlr-gamma-control protocol order: RED, GREEN, BLUE
    // This matches wlsunset's layout: r = table, g = table + ramp_size, b = table + 2*ramp_size
    let mut gamma_data = Vec::with_capacity(size * 3 * 2);

    // Red channel
    for value in red_table {
        gamma_data.extend_from_slice(&value.to_le_bytes());
    }

    // Green channel
    for value in green_table {
        gamma_data.extend_from_slice(&value.to_le_bytes());
    }

    // Blue channel
    for value in blue_table {
        gamma_data.extend_from_slice(&value.to_le_bytes());
    }

    Ok(gamma_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_to_rgb_daylight() {
        // TODO: potencially incorrect for different illuminants
        let (r, g, b) = temperature_to_rgb(6500, 1.0);
        // Daylight should be neutral
        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_temperature_to_rgb_warm() {
        let (r, g, b) = temperature_to_rgb(3300, 1.0);
        // Warm light should be red-heavy, blue-light
        assert!(r > g);
        assert!(g > b);
        assert!(b < 0.8);
    }

    #[test]
    fn test_temperature_to_rgb_cool() {
        let (r, g, b) = temperature_to_rgb(8000, 1.0);
        // Cool light should be blue-heavy
        assert!(b > g);
        assert!(r < b);
    }

    #[test]
    fn test_gamma_table_generation() {
        let table = generate_gamma_table(256, 1.0, 1.0);
        assert_eq!(table.len(), 256);
        assert_eq!(table[0], 0);
        assert_eq!(table[255], 65535);

        // Should be monotonically increasing
        for i in 1..table.len() {
            assert!(table[i] >= table[i - 1]);
        }
    }

    #[test]
    fn test_create_gamma_tables() {
        let tables = create_gamma_tables(256, 6500, 1.0, false).unwrap();
        // Should contain 3 channels * 256 entries * 2 bytes each
        assert_eq!(tables.len(), 256 * 3 * 2);
    }
}
