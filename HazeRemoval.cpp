#include "dehaze.h"
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

            // Get dark channel
            int patch_radius = 7; // Consistent patch radius across all implementations
            ImageF64 darkchannel = zeros<double>(img_double.width, img_double.height);

            for (int i = 0; i < darkchannel.height; i++) {
                for (int j = 0; j < darkchannel.width; j++) {
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(darkchannel.height, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(darkchannel.width, j + patch_radius + 1);
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

                    darkchannel.at(i, j) = dark;
                }
            }

            // End timing for dark channel calculation
            auto darkChannelEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(
                darkChannelEndTime - darkChannelStartTime).count();

            // Start timing for atmospheric light estimation
            auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

            // Get atmospheric light
            int pixels = img_double.height * img_double.width;
            // Ensure at least 1 pixel is used
            int num = std::max(1, pixels / 1000); // 0.1% brightest pixels
            std::vector<std::pair<double, int>> V;
            V.reserve(pixels);

            for (int i = 0; i < darkchannel.height; i++) {
                for (int j = 0; j < darkchannel.width; j++) {
                    int index = i * darkchannel.width + j;
                    V.emplace_back(darkchannel.at(i, j), index);
                }
            }

            std::sort(V.begin(), V.end(), [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.first > p2.first;
                });

            double atmospheric_light[3] = { 0, 0, 0 };
            double maxAtmospheric = 0.0;

            // First pass - find maximum intensity
            for (int k = 0; k < num; k++) {
                int idx = V[k].second;
                int y = idx / darkchannel.width;
                int x = idx % darkchannel.width;

                double avgIntensity = (img_double.at(y, x, 0) + img_double.at(y, x, 1) + img_double.at(y, x, 2)) / 3.0;
                if (avgIntensity > maxAtmospheric) {
                    maxAtmospheric = avgIntensity;
                }
            }

            // Second pass - use only reasonably bright pixels
            int validPixels = 0;
            for (int k = 0; k < num; k++) {
                int idx = V[k].second;
                int y = idx / darkchannel.width;
                int x = idx % darkchannel.width;

                double avgIntensity = (img_double.at(y, x, 0) + img_double.at(y, x, 1) + img_double.at(y, x, 2)) / 3.0;
                if (avgIntensity > maxAtmospheric * 0.7) { // Consistent threshold across implementations
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

            // Use consistent bounds for atmospheric light values
            for (int i = 0; i < 3; i++) {
                atmospheric_light[i] = std::max(0.05, std::min(0.95, atmospheric_light[i]));
            }

            // Check if this is likely an indoor scene (consistent threshold)
            bool isIndoorScene = false;
            double avgAtmospheric = (atmospheric_light[0] + atmospheric_light[1] + atmospheric_light[2]) / 3.0;
            if (avgAtmospheric < 0.6) {
                isIndoorScene = true;
            }

            // End timing for atmospheric light estimation
            auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(
                atmosphericEndTime - atmosphericStartTime).count();

            // Start timing for transmission estimation
            auto transmissionStartTime = std::chrono::high_resolution_clock::now();

            // Adjust omega for indoor/outdoor scenes
            double omega = isIndoorScene ? 0.75 : 0.95; // Consistent omega values

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
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(temp.height, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(temp.width, j + patch_radius + 1);
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

            // Get transmission
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

            // Apply guided filter for refinement - use consistent parameters across implementations
            transmission = fastGuidedFilter(img_double, transmission, 40, 0.1, 4);

            // End timing for refinement
            auto refinementEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.refinementTime = std::chrono::duration<double, std::milli>(
                refinementEndTime - refinementStartTime).count();

            // Start timing for scene reconstruction
            auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

            // Adjust minimum transmission value based on scene type (consistent across implementations)
            double t0 = isIndoorScene ? 0.2 : 0.1;

            // Get a fresh copy of channels
            split(img_double, channels, 3);

            // Create bounded transmission map with better sky region handling
            ImageF64 trans_bounded(transmission.width, transmission.height, 1);

            for (int i = 0; i < transmission.height; i++) {
                for (int j = 0; j < transmission.width; j++) {
                    // Apply minimum transmission threshold
                    trans_bounded.at(i, j) = std::max(transmission.at(i, j), t0);

                    // Special handling for sky regions (typically in top portion of image)
                    if (i < transmission.height / 3) {
                        // Get color info to detect sky
                        double b = img_double.at(i, j, 0); // Blue
                        double g = img_double.at(i, j, 1); // Green
                        double r = img_double.at(i, j, 2); // Red

                        // Improved sky detection - consistent across implementations
                        if ((b > 0.6 && b > r && b > g) ||      // Blue-dominant sky
                            (b > 0.6 && g > 0.6 && r > 0.6)) {  // Bright sky (any color)
                            // Use higher transmission for sky
                            trans_bounded.at(i, j) = std::max(trans_bounded.at(i, j), 0.7);
                        }
                    }
                }
            }

            // Create result image
            ImageF64 result(img_double.width, img_double.height, img_double.channels);

            // Recover the scene
            for (int i = 0; i < img_double.height; i++) {
                for (int j = 0; j < img_double.width; j++) {
                    double t = trans_bounded.at(i, j);

                    // Calculate temporary storage for color values
                    double recovered[3];
                    double lum = 0.0;

                    // Process each channel
                    for (int c = 0; c < 3; c++) {
                        double normalized = img_double.at(i, j, c);
                        // Apply dehaze formula J = (I-A)/t + A with bounds checking
                        recovered[c] = ((normalized - atmospheric_light[c]) / t) + atmospheric_light[c];

                        // Calculate luminance for saturation correction
                        if (c == 0) lum += 0.114 * recovered[c];      // B
                        else if (c == 1) lum += 0.587 * recovered[c]; // G
                        else lum += 0.299 * recovered[c];             // R
                    }

                    // Apply consistent saturation/color correction across all implementations
                    for (int c = 0; c < 3; c++) {
                        // Apply correction for extreme values to reduce artifacts
                        if (recovered[c] > 0.8 || recovered[c] < 0.2) {
                            recovered[c] = recovered[c] * 0.85 + lum * 0.15;
                        }

                        // Apply consistent brightness adjustments based on scene type
                        if (lum < 0.5) {
                            // For darker scenes (like indoor scenes), slightly increase brightness
                            recovered[c] = pow(recovered[c], 0.9);
                        }
                        else {
                            // For brighter scenes, keep as is or slightly reduce brightness
                            recovered[c] = pow(recovered[c], 1.05);
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