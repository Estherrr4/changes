#pragma once
#ifndef DEHAZE_COMMON_H
#define DEHAZE_COMMON_H

// Common parameters for all implementations to ensure consistent results
// Add this file to your project and include it in all dehazing implementations

// Dark channel calculation
#define PATCH_RADIUS 7                // For dark channel calculation

// Guided filter parameters
#define GUIDED_FILTER_RADIUS 40       // For transmission refinement
#define GUIDED_FILTER_EPSILON 0.1     // For transmission refinement
#define GUIDED_FILTER_SUBSAMPLE 4     // For guided filter downsampling

// Atmospheric light estimation
#define ATMOSPHERIC_LIGHT_PERCENTAGE 0.001  // Top 0.1% brightest pixels
#define ATMOSPHERIC_LIGHT_THRESHOLD 0.7     // 70% of max intensity threshold
#define ATMOSPHERIC_LIGHT_MIN 0.05          // Min value bound
#define ATMOSPHERIC_LIGHT_MAX 0.95          // Max value bound

// Scene type detection
#define INDOOR_THRESHOLD 0.6          // Average atmospheric light below this = indoor
#define OMEGA_INDOOR 0.75             // Omega for indoor scenes
#define OMEGA_OUTDOOR 0.95            // Omega for outdoor scenes
#define T0_INDOOR 0.2                 // Minimum transmission for indoor scenes
#define T0_OUTDOOR 0.1                // Minimum transmission for outdoor scenes

// Sky detection parameters
#define SKY_REGION_HEIGHT_RATIO 3     // Process top 1/3 of image for sky
#define BLUE_THRESHOLD 0.6            // Blue channel threshold for sky
#define BRIGHT_THRESHOLD 0.6          // Overall brightness threshold for sky
#define SKY_TRANSMISSION_MIN 0.7      // Minimum transmission for sky

// Color correction parameters
#define EXTREME_VALUE_UPPER 0.8       // Upper threshold for color correction
#define EXTREME_VALUE_LOWER 0.2       // Lower threshold for color correction
#define COLOR_BLEND_FACTOR 0.85       // Blending factor for extreme values
#define DARK_SCENE_THRESHOLD 0.5      // Threshold for dark/bright scene detection
#define DARK_SCENE_GAMMA 0.9          // Gamma for darker scenes
#define BRIGHT_SCENE_GAMMA 1.05       // Gamma for brighter scenes

// Luminance weights (standard RGB to grayscale conversion)
#define LUMINANCE_B 0.114
#define LUMINANCE_G 0.587
#define LUMINANCE_R 0.299

#endif // DEHAZE_COMMON_H
