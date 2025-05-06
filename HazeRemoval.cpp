#include "dehaze.h"
#include "dehaze_common.h"
#include "fastguidedfilter.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <cmath>

namespace DarkChannel {
    // Global timing info that can be accessed from outside
    TimingInfo lastTimingInfo;

    TimingInfo getLastTimingInfo() {
        return lastTimingInfo;
    }

    // Convert OpenCV Mat to custom image format (only for IO purposes)
    ImageU8 matToImage(const cv::Mat& mat) {
        if (mat.empty()) {
            return ImageU8();
        }

        int width = mat.cols;
        int height = mat.rows;
        int channels = mat.channels();

        ImageU8 result(width, height, channels);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (channels == 1) {
                    result.at(y, x, 0) = mat.at<unsigned char>(y, x);
                }
                else if (channels == 3) {
                    const cv::Vec3b& pixel = mat.at<cv::Vec3b>(y, x);
                    result.at(y, x, 0) = pixel[0]; // B
                    result.at(y, x, 1) = pixel[1]; // G
                    result.at(y, x, 2) = pixel[2]; // R
                }
            }
        }
        return result;
    }

    // Convert custom image format back to OpenCV Mat (only for IO purposes)
    cv::Mat imageToMat(const ImageU8& img) {
        if (img.empty()) {
            return cv::Mat();
        }

        cv::Mat result;
        if (img.channels == 1) {
            result = cv::Mat(img.height, img.width, CV_8UC1);
            for (int y = 0; y < img.height; y++) {
                for (int x = 0; x < img.width; x++) {
                    result.at<unsigned char>(y, x) = img.at(y, x, 0);
                }
            }
        }
        else if (img.channels == 3) {
            result = cv::Mat(img.height, img.width, CV_8UC3);
            for (int y = 0; y < img.height; y++) {
                for (int x = 0; x < img.width; x++) {
                    cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
                    pixel[0] = img.at(y, x, 0); // B
                    pixel[1] = img.at(y, x, 1); // G
                    pixel[2] = img.at(y, x, 2); // R
                }
            }
        }
        return result;
    }

    // Serial implementation
    ImageU8 dehaze(const ImageU8& img) {
        try {
            // Reset timing info
            lastTimingInfo = TimingInfo();

            // Start total time measurement
            auto totalStartTime = std::chrono::high_resolution_clock::now();

            // Check if image is valid
            if (img.empty()) {
                std::cerr << "Error: Input image is empty" << std::endl;
                return img;
            }

            // Check for correct image type
            if (img.channels != 3) {
                std::cerr << "Error: Only 3-channel images are supported" << std::endl;
                return img;
            }

            // Convert to double precision for calculations
            ImageF64 img_double = img.convertTo<double>(1.0 / 255.0);

            // Start timing for dark channel calculation
            auto darkChannelStartTime = std::chrono::high_resolution_clock::now();

            //------------------------------------------------------------------
            // Step 1: Get dark channel using consistent parameters
            //------------------------------------------------------------------
            ImageF64 darkChannel = zeros<double>(img_double.width, img_double.height);

            for (int i = 0; i < darkChannel.height; i++) {
                for (int j = 0; j < darkChannel.width; j++) {
                    int r_start = std::max(0, i - PATCH_RADIUS);
                    int r_end = std::min(darkChannel.height, i + PATCH_RADIUS + 1);
                    int c_start = std::max(0, j - PATCH_RADIUS);
                    int c_end = std::min(darkChannel.width, j + PATCH_RADIUS + 1);
                    double dark = 1.0;  // Initialize to max normalized value (1.0)

                    for (int r = r_start; r < r_end; r++) {
                        for (int c = c_start; c < c_end; c++) {
                            double val0 = img_double.at(r, c, 0); // B
                            double val1 = img_double.at(r, c, 1); // G
                            double val2 = img_double.at(r, c, 2); // R

                            dark = std::min(dark, val0);
                            dark = std::min(dark, val1);
                            dark = std::min(dark, val2);
                        }
                    }

                    darkChannel.at(i, j) = dark;
                }
            }

            // End timing for dark channel calculation
            auto darkChannelEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(
                darkChannelEndTime - darkChannelStartTime).count();

            // Start timing for atmospheric light estimation
            auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

            //------------------------------------------------------------------
            // Step 2: Estimate atmospheric light using consistent algorithm
            //------------------------------------------------------------------

            // Get atmospheric light
            int pixels = img_double.height * img_double.width;
            // Ensure at least 1 pixel is used
            int numBrightestPixels = std::max(1, static_cast<int>(pixels * ATMOSPHERIC_LIGHT_PERCENTAGE));
            std::vector<std::pair<double, int>> V;
            V.reserve(pixels);

            for (int i = 0; i < darkChannel.height; i++) {
                for (int j = 0; j < darkChannel.width; j++) {
                    int index = i * darkChannel.width + j;
                    V.emplace_back(darkChannel.at(i, j), index);
                }
            }

            std::sort(V.begin(), V.end(), [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.first > p2.first;
                });

            double atmospheric_light[3] = { 0, 0, 0 };
            double maxAtmospheric = 0.0;

            // First pass - find maximum intensity
            for (int k = 0; k < numBrightestPixels; k++) {
                int idx = V[k].second;
                int y = idx / darkChannel.width;
                int x = idx % darkChannel.width;

                double avgIntensity = (img_double.at(y, x, 0) + img_double.at(y, x, 1) + img_double.at(y, x, 2)) / 3.0;
                maxAtmospheric = std::max(maxAtmospheric, avgIntensity);
            }

            // Second pass - use only reasonably bright pixels
            int validPixels = 0;
            for (int k = 0; k < numBrightestPixels; k++) {
                int idx = V[k].second;
                int y = idx / darkChannel.width;
                int x = idx % darkChannel.width;

                double avgIntensity = (img_double.at(y, x, 0) + img_double.at(y, x, 1) + img_double.at(y, x, 2)) / 3.0;

                // Only include pixels that are at least ATMOSPHERIC_LIGHT_THRESHOLD of max brightness
                if (avgIntensity > maxAtmospheric * ATMOSPHERIC_LIGHT_THRESHOLD) {
                    atmospheric_light[0] += img_double.at(y, x, 0);
                    atmospheric_light[1] += img_double.at(y, x, 1);
                    atmospheric_light[2] += img_double.at(y, x, 2);
                    validPixels++;
                }
            }

            if (validPixels > 0) {
                atmospheric_light[0] /= validPixels;
                atmospheric_light[1] /= validPixels;
                atmospheric_light[2] /= validPixels;
            }
            else {
                // Fallback if no good pixels found
                atmospheric_light[0] = 0.8;
                atmospheric_light[1] = 0.8;
                atmospheric_light[2] = 0.8;
            }

            // Apply consistent bounds for atmospheric light values
            for (int i = 0; i < 3; i++) {
                atmospheric_light[i] = std::max(ATMOSPHERIC_LIGHT_MIN, std::min(ATMOSPHERIC_LIGHT_MAX, atmospheric_light[i]));
            }

            // Check if this is likely an indoor scene - consistent detection logic
            bool isIndoorScene = false;
            double avgAtmospheric = (atmospheric_light[0] + atmospheric_light[1] + atmospheric_light[2]) / 3.0;
            if (avgAtmospheric < INDOOR_THRESHOLD) {
                isIndoorScene = true;
            }

            // End timing for atmospheric light estimation
            auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(
                atmosphericEndTime - atmosphericStartTime).count();

            // Start timing for transmission estimation
            auto transmissionStartTime = std::chrono::high_resolution_clock::now();

            //------------------------------------------------------------------
            // Step 3: Calculate transmission map using consistent algorithm
            //------------------------------------------------------------------

            // Adjust omega based on scene type using consistent parameters
            double omega = isIndoorScene ? OMEGA_INDOOR : OMEGA_OUTDOOR;

            ImageF64 channels[3];
            split(img_double, channels, 3);

            for (int k = 0; k < 3; k++) {
                for (int i = 0; i < channels[k].height; i++) {
                    for (int j = 0; j < channels[k].width; j++) {
                        channels[k].at(i, j) /= atmospheric_light[k];
                    }
                }
            }

            ImageF64 temp;
            merge(channels, 3, temp);

            // Get dark channel of normalized image
            ImageF64 temp_dark = zeros<double>(temp.width, temp.height);

            for (int i = 0; i < temp.height; i++) {
                for (int j = 0; j < temp.width; j++) {
                    int r_start = std::max(0, i - PATCH_RADIUS);
                    int r_end = std::min(temp.height, i + PATCH_RADIUS + 1);
                    int c_start = std::max(0, j - PATCH_RADIUS);
                    int c_end = std::min(temp.width, j + PATCH_RADIUS + 1);
                    double dark = 1.0;

                    for (int r = r_start; r < r_end; r++) {
                        for (int c = c_start; c < c_end; c++) {
                            double val0 = temp.at(r, c, 0);
                            double val1 = temp.at(r, c, 1);
                            double val2 = temp.at(r, c, 2);

                            dark = std::min(dark, val0);
                            dark = std::min(dark, val1);
                            dark = std::min(dark, val2);
                        }
                    }

                    temp_dark.at(i, j) = dark;
                }
            }

            // Calculate transmission map
            ImageF64 transmission(temp_dark.width, temp_dark.height, 1);

            for (int i = 0; i < transmission.height; i++) {
                for (int j = 0; j < transmission.width; j++) {
                    transmission.at(i, j) = 1.0 - omega * temp_dark.at(i, j);
                }
            }

            // End timing for transmission estimation
            auto transmissionEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.transmissionTime = std::chrono::duration<double, std::milli>(
                transmissionEndTime - transmissionStartTime).count();

            // Start timing for refinement
            auto refinementStartTime = std::chrono::high_resolution_clock::now();

            //------------------------------------------------------------------
            // Step 4: Refine transmission using guided filter with consistent parameters
            //------------------------------------------------------------------

            // Apply guided filter for refinement with consistent parameters
            transmission = fastGuidedFilter(img_double, transmission,
                GUIDED_FILTER_RADIUS,
                GUIDED_FILTER_EPSILON,
                GUIDED_FILTER_SUBSAMPLE);

            // Apply sky region handling - consistent across all implementations
            int skyRegionHeight = transmission.height / SKY_REGION_HEIGHT_RATIO;

            for (int i = 0; i < skyRegionHeight; i++) {
                for (int j = 0; j < transmission.width; j++) {
                    // Get color info to detect sky
                    double b = img_double.at(i, j, 0); // Blue
                    double g = img_double.at(i, j, 1); // Green
                    double r = img_double.at(i, j, 2); // Red

                    // Sky detection using consistent criteria
                    bool isSky = false;

                    // Blue-dominant sky detection
                    if (b > BLUE_THRESHOLD && b > r && b > g) {
                        isSky = true;
                    }

                    // Bright sky detection (any color)
                    if (b > BRIGHT_THRESHOLD && g > BRIGHT_THRESHOLD && r > BRIGHT_THRESHOLD) {
                        isSky = true;
                    }

                    // Apply consistent transmission adjustment for sky
                    if (isSky) {
                        transmission.at(i, j) = std::max(transmission.at(i, j), SKY_TRANSMISSION_MIN);
                    }
                }
            }

            // End timing for refinement
            auto refinementEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.refinementTime = std::chrono::duration<double, std::milli>(
                refinementEndTime - refinementStartTime).count();

            // Start timing for scene reconstruction
            auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

            //------------------------------------------------------------------
            // Step 5: Reconstruct scene using consistent algorithm and color correction
            //------------------------------------------------------------------

            // Use consistent transmission minimum threshold based on scene type
            double t0 = isIndoorScene ? T0_INDOOR : T0_OUTDOOR;

            // Create result image
            ImageF64 result(img_double.width, img_double.height, img_double.channels);

            // Recover the scene
            for (int i = 0; i < img_double.height; i++) {
                for (int j = 0; j < img_double.width; j++) {
                    // Apply minimum transmission threshold
                    double t = std::max(transmission.at(i, j), t0);

                    // Temporary storage for color values
                    double recovered[3];
                    double lum = 0.0;

                    // Process each channel
                    for (int c = 0; c < 3; c++) {
                        double normalized = img_double.at(i, j, c);
                        // Apply dehaze formula J = (I-A)/t + A
                        recovered[c] = ((normalized - atmospheric_light[c]) / t) + atmospheric_light[c];

                        // Calculate luminance for saturation correction
                        // Using standard RGB to grayscale weights
                        if (c == 0) lum += LUMINANCE_B * recovered[c];      // B
                        else if (c == 1) lum += LUMINANCE_G * recovered[c]; // G
                        else lum += LUMINANCE_R * recovered[c];             // R
                    }

                    // Apply consistent saturation/color correction
                    for (int c = 0; c < 3; c++) {
                        // Correction for extreme values to reduce artifacts
                        if (recovered[c] > EXTREME_VALUE_UPPER || recovered[c] < EXTREME_VALUE_LOWER) {
                            recovered[c] = recovered[c] * COLOR_BLEND_FACTOR + lum * (1.0 - COLOR_BLEND_FACTOR);
                        }

                        // Apply brightness adjustments based on luminance
                        if (lum < DARK_SCENE_THRESHOLD) {
                            // For darker scenes, slightly increase brightness
                            recovered[c] = std::pow(recovered[c], DARK_SCENE_GAMMA);
                        }
                        else {
                            // For brighter scenes, slightly reduce brightness
                            recovered[c] = std::pow(recovered[c], BRIGHT_SCENE_GAMMA);
                        }

                        // Ensure bounds and store
                        result.at(i, j, c) = std::max(0.0, std::min(1.0, recovered[c]));
                    }
                }
            }

            // Convert back to 8-bit
            ImageU8 result_8bit = result.convertTo<unsigned char>(255.0);

            // End timing for scene reconstruction
            auto reconstructionEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.reconstructionTime = std::chrono::duration<double, std::milli>(
                reconstructionEndTime - reconstructionStartTime).count();

            // End total time measurement
            auto totalEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.totalTime = std::chrono::duration<double, std::milli>(
                totalEndTime - totalStartTime).count();

            std::cout << "\n===== Serial Performance Timing (milliseconds) =====" << std::endl;
            std::cout << std::fixed << std::setprecision(10);
            std::cout << "Dark Channel Calculation: " << lastTimingInfo.darkChannelTime << " ms" << std::endl;
            std::cout << "Atmospheric Light Estimation: " << lastTimingInfo.atmosphericLightTime << " ms" << std::endl;
            std::cout << "Transmission Estimation: " << lastTimingInfo.transmissionTime << " ms" << std::endl;
            std::cout << "Transmission Refinement: " << lastTimingInfo.refinementTime << " ms" << std::endl;
            std::cout << "Scene Reconstruction: " << lastTimingInfo.reconstructionTime << " ms" << std::endl;
            std::cout << "Total Execution Time: " << lastTimingInfo.totalTime << " ms" << std::endl;
            std::cout << "========================================" << std::endl;

            return result_8bit;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in serial implementation: " << e.what() << std::endl;
            return img;
        }
        catch (...) {
            std::cerr << "Unknown exception in dehaze function" << std::endl;
            return img;
        }
    }
}