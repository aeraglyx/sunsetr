//! Color temperature and brightness to RGB conversion.
//!
//! This module provides efficient colorimetric calculations for converting
//! color temperatures to RGB values for gamma table generation.
//!

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

/// Apply Opto-Electronic Transfer Function
/// https://en.wikipedia.org/wiki/Transfer_functions_in_imaging
fn apply_oetf(rgb: &mut Rgb, oetf: fn(f64) -> f64) {
    rgb.r = oetf(rgb.r);
    rgb.g = oetf(rgb.g);
    rgb.b = oetf(rgb.b);
}

const MATRIX_XYZ_TO_REC709: [f64; 9] = [
    3.2410032, -1.5373990, -0.4986159,
    -0.9692243, 1.8759300, 0.0415542,
    0.0556394, -0.2040112, 1.0571490,
];

/// "wide"
const MATRIX_XYZ_TO_REC2020: [f64; 9] = [
    1.7166634, -0.3556733, -0.2533681,
    -0.6666738, 1.6164557, 0.0157683,
    0.0176425, -0.0427770, 0.9422433,
];

/// P3 + 6300 K
const MATRIX_XYZ_TO_DCI_P3: [f64; 9] = [
    2.7253940, -1.0180030, -0.4401632,
    -0.7951680, 1.6897321, 0.0226472,
    0.0412419, -0.0876390, 1.1009294,
];

/// P3 + 6500 K
const MATRIX_XYZ_TO_DISPLAY_P3: [f64; 9] = [
    2.4935091, -0.9313882, -0.4027128,
    -0.8294732, 1.7626306, 0.0236242,
    0.0358513, -0.0761839, 0.9570296,
];

/// "a98"
const MATRIX_XYZ_TO_ADOBE: [f64; 9] = [
    2.0415913, -0.5650079, -0.3447319,
    -0.9692243, 1.8759300, 0.0415542,
    0.0134464, -0.1183813, 1.0153376,
];

/// Convert XYZ color space to RGB using standard transformation matrix
/// Reference: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
/// Reference: https://observablehq.com/@danburzo/color-matrix-calculator
fn xyz_to_rgb(xyz: &Xyz, matrix: [f64; 9]) -> Rgb {
    Rgb {
        r: matrix[0] * xyz.x + matrix[1] * xyz.y + matrix[2] * xyz.z,
        g: matrix[3] * xyz.x + matrix[4] * xyz.y + matrix[5] * xyz.z,
        b: matrix[6] * xyz.x + matrix[7] * xyz.y + matrix[8] * xyz.z,
    }
}

/// Adapted from cubic polynomial smooth-minimum by Inigo Quilez
/// Reference: https://iquilezles.org/articles/smin/
fn smooth_max(a: f64, b: f64, falloff: f64) -> f64 {
    let k = 6.0 * falloff;
    let h = (k - (a - b).abs()).max(0.0) / k;
    (a).max(b) + k * h.powi(3) / 6.0
}

/// Compress gamut for softer falloff at the gamut boundary
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

/// Reference: https://en.wikipedia.org/wiki/Smoothstep
fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    3.0 * x.powi(2) - 2.0 * x.powi(3)
}

/// Convert temperature from Kelvins to Mireds
fn k_to_mired(kelvin: u32) -> f64 {
    1000000.0 / kelvin as f64
}

/// https://wiki.hypr.land/Configuring/Monitors/#color-management-presets
enum ColorSpace {
    SRGB,
    Wide,
    DP3,
    DciP3,
    Adobe,
}

enum Illuminant {
    D63, // technacally a wrong name
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

fn get_xyz_to_rgb_matrix(color_space: &ColorSpace) -> [f64; 9] {
    match color_space {
        ColorSpace::SRGB => MATRIX_XYZ_TO_REC709,
        ColorSpace::Wide => MATRIX_XYZ_TO_REC2020,
        ColorSpace::DciP3 => MATRIX_XYZ_TO_DCI_P3,
        ColorSpace::DP3 => MATRIX_XYZ_TO_DISPLAY_P3,
        ColorSpace::Adobe => MATRIX_XYZ_TO_ADOBE,
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
        Illuminant::D63 => Xy { x: 0.31400, y: 0.35100 },
        Illuminant::D65 => Xy { x: 0.31271, y: 0.32902 },
    }
}

/// Calculate RGB values for a given color temperature
///
/// This algorithm provides accurate color temperature to RGB conversion from
/// 1000K to 20000K. The algorithm approximates Plackian locus XY coordinates, applies
/// a small offset to match the whitepoint exactly and converts XY to normalized RGB.
///
/// # Arguments
/// * `temp` - Color temperature in Kelvin (1000-20000)
///
/// # Returns
/// Rgb in range 0.0-1.0 with f64 precision
fn temperature_to_rgb(temp: u32, color_space: &ColorSpace) -> Rgb {
    let illuminant = get_illuminant(&color_space);

    let expected_wp_temp = get_wp_cct(&illuminant);
    let expected_wp_chroma = get_wp_chroma(&illuminant);

    if temp == expected_wp_temp {
        return Rgb { r: 1.0, g: 1.0, b: 1.0 };
    }

    let chroma_at_wp = temperature_to_chroma(expected_wp_temp as f64);
    let chroma_at_temp = temperature_to_chroma(temp as f64);

    // Offset the locus so it intersects the monitor's whitepoint exactly.
    // That way, when a user sets 6500K on a D65 monitor, they'll get "true white"
    // but more extreme temperatures will blend into the Planckian locus.
    const FALLOFF: f64 = 150.0;
    let offset_weight = smoothstep(
        1.0 - (k_to_mired(temp) - k_to_mired(expected_wp_temp)).abs() / FALLOFF
    );

    // Construct chromaticity coordinates
    // TODO: offset could be precomputed
    let wp = Xy {
        x: chroma_at_temp.x + offset_weight * (expected_wp_chroma.x - chroma_at_wp.x),
        y: chroma_at_temp.y + offset_weight * (expected_wp_chroma.y - chroma_at_wp.y),
    };

    // Convert chromaticity coordinates to XYZ
    // Ignoring overall luminance for performance, since we later normalize.
    let wp_z = 1.0 - wp.x - wp.y;
    let xyz = Xyz {
        x: wp.x,
        y: wp.y,
        z: wp_z,
    };

    // Convert XYZ to RGB
    let matrix = get_xyz_to_rgb_matrix(&color_space);
    let mut rgb = xyz_to_rgb(&xyz, matrix);

    // Normalize and apply gamut compression
    rgb_normalize(&mut rgb);
    gamut_compression(&mut rgb, 0.005);

    rgb
}

/// Calculate RGB values for a given color temperature and brightness
///
/// This algorithm provides an RGB multiplier to achieve a certain color temperature
/// and brightness.
///
/// # Arguments
/// * `temp` - Color temperature in Kelvin (1000-20000)
/// * `brightness` - Brightness multiplier (0.1-2.0)
///
/// # Returns
/// Tuple of (red, green, blue) factors in range 0.0-1.0 with f64 precision
pub fn state_to_rgb(temp: u32, brightness: f64) -> (f64, f64, f64) {
    // TODO: pass color space
    let color_space = ColorSpace::SRGB;

    let mut rgb = temperature_to_rgb(temp, &color_space);

    // Apply brightness
    rgb_brightness(&mut rgb, brightness.powi(3));

    // Apply Opto-Electronic Transfer Function to compensate for compositor limitations
    let oetf = get_oetf(&color_space);
    apply_oetf(&mut rgb, oetf);

    (rgb.r, rgb.g, rgb.b)
}

/// Get RGB factors for a given color temperature as a formatted tuple.
///
/// This is a convenience function for debug logging. Values are rounded
/// to 3 decimal places for display purposes only.
///
/// # Arguments
/// * `temperature` - Color temperature in Kelvin (1000-20000)
///
/// # Returns
/// Tuple of (red, green, blue) factors rounded to 3 decimal places
pub fn get_rgb_factors(temperature: u32) -> (f64, f64, f64) {
    let rgb = temperature_to_rgb(temperature, &ColorSpace::SRGB);
    // Round to 3 decimal places for cleaner logging
    (
        (rgb.r * 1000.0).round() / 1000.0,
        (rgb.g * 1000.0).round() / 1000.0,
        (rgb.b * 1000.0).round() / 1000.0,
    )
}

/// Generate gamma table for a specific color channel.
///
/// Creates a gamma lookup table (LUT) that maps input values to output values
/// using a power function gamma curve.
///
/// The formula applied is: output = (input * color_factor)^(1/gamma)
/// where:
/// - input is normalized 0.0-1.0
/// - color_factor adjusts for color temperature (0.0-1.0)
/// - gamma controls the brightness curve (typically 0.9-1.0)
/// - output is scaled to 0-65535 for 16-bit protocol
///
/// # Arguments
/// * `size` - Size of the gamma table (typically 256 or 1024)
/// * `color_factor` - Color temperature adjustment factor (0.0-1.0)
/// * `gamma` - Gamma curve value (0.9 = 90% brightness, 1.0 = 100%)
///
/// # Returns
/// Vector of 16-bit gamma values for this color channel
pub fn generate_gamma_table(size: usize, color_factor: f64) -> Vec<u16> {
    let mut table = Vec::with_capacity(size);

    for i in 0..size {
        // Calculate normalized input value (0.0 to 1.0) with f64 precision
        let val = i as f64 / (size - 1) as f64;

        // Apply color temperature factor
        // Maintain f64 precision throughout calculation to minimize rounding errors
        let output = (val * color_factor * 65535.0).clamp(0.0, 65535.0);

        // Convert to u16 only at final step (required by protocol)
        table.push(output as u16);
    }

    table
}

/// Create complete gamma tables for RGB channels.
///
/// Generates the full set of gamma lookup tables needed for the
/// wlr-gamma-control-unstable-v1 protocol. Uses f64 precision internally
/// to minimize quantization artifacts in the final u16 output.
///
/// # Arguments
/// * `size` - Size of each gamma table (reported by compositor)
/// * `temperature` - Color temperature in Kelvin (1000-20000)
/// * `gamma_percent` - Gamma adjustment as decimal (0.9 = 90%, 1.0 = 100%)
/// * `debug_enabled` - Whether to output debug information
///
/// # Returns
/// Byte vector containing concatenated R, G, B gamma tables in little-endian format
pub fn create_gamma_tables(
    size: usize,
    temperature: u32,
    gamma_percent: f64,
    debug_enabled: bool,
) -> Result<Vec<u8>> {
    // Calculate RGB factors with maximum precision
    let (red_factor, green_factor, blue_factor) = state_to_rgb(temperature, gamma_percent / 100.0);

    // Generate individual channel tables using f64 precision throughout
    // Only convert to u16 at the final step in generate_gamma_table
    let red_table = generate_gamma_table(size, red_factor);
    let green_table = generate_gamma_table(size, green_factor);
    let blue_table = generate_gamma_table(size, blue_factor);

    // Log sample values for debugging
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
    // Protocol order: RED, GREEN, BLUE as documented in wlr-gamma-control
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
        let (r, g, b) = state_to_rgb(6500, 1.0);
        // Daylight should be approximately neutral
        // TODO: potentially incorrect for different illuminants
        assert!((r - 1.0).abs() < 0.01);
        assert!((g - 1.0).abs() < 0.01);
        assert!((b - 1.0).abs() < 0.01); // Blue is slightly lower in the algorithm
    }

    #[test]
    fn test_temperature_to_rgb_warm() {
        let (r, g, b) = state_to_rgb(3300, 1.0);
        // Warm light should be red-heavy, blue-light
        assert!(r > g);
        assert!(g > b);
        assert!(b < 0.8);
    }

    #[test]
    fn test_temperature_to_rgb_cool() {
        let (r, g, b) = state_to_rgb(8000, 1.0);
        // Cool light should be blue-heavy
        assert!(b > g);
        assert!(r < b);
    }

    #[test]
    fn test_temperature_to_rgb_very_warm() {
        let (r, g, b) = state_to_rgb(2000, 1.0);
        // Very warm temperatures should have low blue
        assert!(r > g);
        assert!(g > b);
        assert!(b < 0.1);
    }

    #[test]
    fn test_gamma_table_generation() {
        let table = generate_gamma_table(256, 1.0);
        assert_eq!(table.len(), 256);
        assert_eq!(table[0], 0);
        assert_eq!(table[255], 65535);

        // Should be monotonically increasing
        for i in 1..table.len() {
            assert!(table[i] >= table[i - 1]);
        }
    }

    #[test]
    fn test_gamma_table_with_color_factor() {
        let full_table = generate_gamma_table(256, 1.0);
        let half_table = generate_gamma_table(256, 0.5);

        // Half color factor should produce lower values
        assert!(half_table[255] < full_table[255]);
        assert!(half_table[255] < 40000); // Should be roughly half
    }

    #[test]
    fn test_create_gamma_tables() {
        let tables = create_gamma_tables(256, 6500, 1.0, false).unwrap();
        // Should contain 3 channels * 256 entries * 2 bytes each
        assert_eq!(tables.len(), 256 * 3 * 2);
    }

    #[test]
    fn test_precision_warm_temperatures() {
        // Test that very close temperatures produce different RGB values
        let (r1, g1, b1) = state_to_rgb(2000, 1.0);
        let (r2, g2, b2) = state_to_rgb(2001, 1.0);

        // Values should be different (not equal due to precision loss)
        assert!(r1 != r2 || g1 != g2 || b1 != b2);
    }
}
