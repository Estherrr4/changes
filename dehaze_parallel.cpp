//#include "dehaze_parallel.h"
//#include "fastguidedfilter.h"
//#include "dehaze.h"
//#include <iostream>
//#include <chrono>
//#include <algorithm>
//#include <vector>
//#include <omp.h>
//#include <iomanip>

/*namespace DarkChannel {
    extern TimingInfo lastTimingInfo;

    ImageU8 dehazeParallel(const ImageU8& img, int numThreads) {
        try {
            // Reset timing info
            TimingInfo timingInfo;

            // Start total time measurement
            auto totalStartTime = std::chrono::high_resolution_clock::now();

            // Set number of threads
            omp_set_num_threads(numThreads);

            std::cout << "\n===== OpenMP Performance Timing =====" << std::endl;

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
            int patch_radius = 7;
            ImageF64 darkchannel = zeros<double>(img_double.width, img_double.height);

            // Use OpenMP for parallelism
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < darkchannel.height; i++) {
                for (int j = 0; j < darkchannel.width; j++) {
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(darkchannel.height, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(darkchannel.width, j + patch_radius + 1);
                    double dark = 1.0; // Start with maximum value (after normalization)

                    for (int r = r_start; r < r_end; r++) {
                        for (int c = c_start; c < c_end; c++) {
                            double val0 = img_double.at(r, c, 0);
                            double val1 = img_double.at(r, c, 1);
                            double val2 = img_double.at(r, c, 2);

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
            timingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(darkChannelEndTime - darkChannelStartTime).count();

            // Start timing for atmospheric light estimation
            auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

            // Get atmospheric light
            int pixels = img_double.height * img_double.width;
            // Ensure at least 1 pixel is used
            int num = std::max(1, pixels / 1000); // 0.1%
            std::vector<std::pair<double, int>> V;
            V.reserve(pixels);

            // Create flat index for each pixel with its dark channel value
            for (int i = 0; i < darkchannel.height; i++) {
                for (int j = 0; j < darkchannel.width; j++) {
                    int index = i * darkchannel.width + j;
                    V.emplace_back(darkchannel.at(i, j), index);
                }
            }

            // Sort to find brightest pixels in dark channel
            std::sort(V.begin(), V.end(), [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.first > p2.first;
                });

            // Initialize atmospheric light array
            double atmospheric_light[3] = { 0, 0, 0 };

            // Calculate atmospheric light from brightest pixels
            for (int k = 0; k < num; k++) {
                int idx = V[k].second;
                int r = idx / darkchannel.width;
                int c = idx % darkchannel.width;

                atmospheric_light[0] += img_double.at(r, c, 0);
                atmospheric_light[1] += img_double.at(r, c, 1);
                atmospheric_light[2] += img_double.at(r, c, 2);
            }

            atmospheric_light[0] /= num;
            atmospheric_light[1] /= num;
            atmospheric_light[2] /= num;

            // Avoid division by zero
            for (int i = 0; i < 3; i++) {
                if (atmospheric_light[i] < 0.00001) {
                    atmospheric_light[i] = 0.00001;
                }
            }

            // End timing for atmospheric light estimation
            auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(atmosphericEndTime - atmosphericStartTime).count();

            // Start timing for transmission estimation
            auto transmissionStartTime = std::chrono::high_resolution_clock::now();

            double omega = 0.95;
            ImageF64 channels[3];
            split(img_double, channels, 3);

            // Normalize each channel by atmospheric light
#pragma omp parallel for
            for (int k = 0; k < 3; k++) {
                for (int i = 0; i < channels[k].height; i++) {
                    for (int j = 0; j < channels[k].width; j++) {
                        channels[k].at(i, j) /= atmospheric_light[k];
                    }
                }
            }

            // Merge normalized channels
            ImageF64 temp;
            merge(channels, 3, temp);

            // Get dark channel of normalized image
            ImageF64 temp_dark = zeros<double>(temp.width, temp.height);

#pragma omp parallel for schedule(dynamic)
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

#pragma omp parallel for collapse(2)
            for (int i = 0; i < transmission.height; i++) {
                for (int j = 0; j < transmission.width; j++) {
                    transmission.at(i, j) = 1.0 - omega * temp_dark.at(i, j);
                }
            }

            // End timing for transmission estimation
            auto transmissionEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.transmissionTime = std::chrono::duration<double, std::milli>(transmissionEndTime - transmissionStartTime).count();

            // Start timing for refinement
            auto refinementStartTime = std::chrono::high_resolution_clock::now();

            // Apply guided filter for refinement
            transmission = fastGuidedFilter(img_double, transmission, 40, 0.1, 5);

            // End timing for refinement
            auto refinementEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.refinementTime = std::chrono::duration<double, std::milli>(refinementEndTime - refinementStartTime).count();

            // Start timing for scene reconstruction
            auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

            // Ensure minimum transmission value to preserve details in dark regions
            double t0 = 0.1;

            // Get a fresh copy of the image channels
            split(img_double, channels, 3);

            // Apply minimum transmission
            ImageF64 trans_bounded(transmission.width, transmission.height, 1);

#pragma omp parallel for collapse(2)
            for (int i = 0; i < trans_bounded.height; i++) {
                for (int j = 0; j < trans_bounded.width; j++) {
                    trans_bounded.at(i, j) = std::max(transmission.at(i, j), t0);
                }
            }

            // Create result image
            ImageF64 result(img_double.width, img_double.height, img_double.channels);

            // Process all pixels in parallel
#pragma omp parallel for collapse(2)
            for (int i = 0; i < img_double.height; i++) {
                for (int j = 0; j < img_double.width; j++) {
                    const double t = trans_bounded.at(i, j);

                    // Apply dehaze formula to each channel: J = (I-A)/t + A
                    for (int c = 0; c < 3; c++) {
                        double dehazed = ((channels[c].at(i, j) - atmospheric_light[c]) / t) + atmospheric_light[c];
                        // Clamp to valid range
                        result.at(i, j, c) = std::max(0.0, std::min(1.0, dehazed));
                    }
                }
            }

            // Convert back to 8-bit
            ImageU8 result_8bit = result.convertTo<unsigned char>(255.0);

            // End timing for scene reconstruction
            auto reconstructionEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.reconstructionTime = std::chrono::duration<double, std::milli>(reconstructionEndTime - reconstructionStartTime).count();

            // End total time measurement
            auto totalEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.totalTime = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();

            std::cout << std::fixed << std::setprecision(10);
            std::cout << "Dark Channel Calculation: " << timingInfo.darkChannelTime << " ms" << std::endl;
            std::cout << "Atmospheric Light Estimation: " << timingInfo.atmosphericLightTime << " ms" << std::endl;
            std::cout << "Transmission Estimation: " << timingInfo.transmissionTime << " ms" << std::endl;
            std::cout << "Transmission Refinement: " << timingInfo.refinementTime << " ms" << std::endl;
            std::cout << "Scene Reconstruction: " << timingInfo.reconstructionTime << " ms" << std::endl;
            std::cout << "Total Execution Time: " << timingInfo.totalTime << " ms" << std::endl;
            std::cout << "========================================" << std::endl;

            lastTimingInfo = timingInfo;

            return result_8bit;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in OpenMP implementation: " << e.what() << std::endl;
            return img;
        }
        catch (...) {
            std::cerr << "Unknown exception in dehazeParallel function" << std::endl;
            return img;
        }
    }
}*/

#include "dehaze_parallel.h"
#include "fastguidedfilter.h"
#include "dehaze.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <iomanip>

namespace DarkChannel {
    extern TimingInfo lastTimingInfo;

    ImageU8 dehazeParallel(const ImageU8& img, int numThreads) {
        try {
            // Reset timing info
            TimingInfo timingInfo;

            // Start total time measurement
            auto totalStartTime = std::chrono::high_resolution_clock::now();

            // Set number of threads
            omp_set_num_threads(numThreads);

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
            int patch_radius = 7; // Consistent with serial and CUDA
            ImageF64 darkchannel = zeros<double>(img_double.width, img_double.height);

            // Use OpenMP for parallelism
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < darkchannel.height; i++) {
                for (int j = 0; j < darkchannel.width; j++) {
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(darkchannel.height, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(darkchannel.width, j + patch_radius + 1);
                    double dark = 1.0; // Start with maximum value (after normalization)

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
            timingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(
                darkChannelEndTime - darkChannelStartTime).count();

            // Start timing for atmospheric light estimation
            auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

            // Get atmospheric light
            int pixels = img_double.height * img_double.width;
            // Ensure at least 1 pixel is used
            int num = std::max(1, pixels / 1000); // 0.1% same as in serial and CUDA
            std::vector<std::pair<double, int>> V;
            V.reserve(pixels);

            // Create flat index for each pixel with its dark channel value
            for (int i = 0; i < darkchannel.height; i++) {
                for (int j = 0; j < darkchannel.width; j++) {
                    int index = i * darkchannel.width + j;
                    V.emplace_back(darkchannel.at(i, j), index);
                }
            }

            // Sort to find brightest pixels in dark channel
            std::sort(V.begin(), V.end(), [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.first > p2.first;
                });

            // Initialize atmospheric light array
            double atmospheric_light[3] = { 0, 0, 0 };
            double maxAtmospheric = 0.0;

            // First pass - find maximum intensity
            for (int k = 0; k < num; k++) {
                int idx = V[k].second;
                int r = idx / darkchannel.width;
                int c = idx % darkchannel.width;

                double avgIntensity = (img_double.at(r, c, 0) + img_double.at(r, c, 1) + img_double.at(r, c, 2)) / 3.0;
                if (avgIntensity > maxAtmospheric) {
                    maxAtmospheric = avgIntensity;
                }
            }

            // Second pass - use only reasonably bright pixels
            int validPixels = 0;
            for (int k = 0; k < num; k++) {
                int idx = V[k].second;
                int r = idx / darkchannel.width;
                int c = idx % darkchannel.width;

                double avgIntensity = (img_double.at(r, c, 0) + img_double.at(r, c, 1) + img_double.at(r, c, 2)) / 3.0;
                if (avgIntensity > maxAtmospheric * 0.7) { // Same threshold as serial and CUDA
                    atmospheric_light[0] += img_double.at(r, c, 0);
                    atmospheric_light[1] += img_double.at(r, c, 1);
                    atmospheric_light[2] += img_double.at(r, c, 2);
                    validPixels++;
                }
            }

            if (validPixels > 0) {
                atmospheric_light[0] /= validPixels;
                atmospheric_light[1] /= validPixels;
                atmospheric_light[2] /= validPixels;
            }
            else {
                // Fallback if no good pixels found - same as in serial and CUDA
                atmospheric_light[0] = 0.8;
                atmospheric_light[1] = 0.8;
                atmospheric_light[2] = 0.8;
            }

            // Use consistent bounds for atmospheric light values
            for (int i = 0; i < 3; i++) {
                atmospheric_light[i] = std::max(0.05, std::min(0.95, atmospheric_light[i]));
            }

            // Check if this is likely an indoor scene - consistent threshold with serial and CUDA
            bool isIndoorScene = false;
            double avgAtmospheric = (atmospheric_light[0] + atmospheric_light[1] + atmospheric_light[2]) / 3.0;
            if (avgAtmospheric < 0.6) {
                isIndoorScene = true;
            }

            // End timing for atmospheric light estimation
            auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(
                atmosphericEndTime - atmosphericStartTime).count();

            // Start timing for transmission estimation
            auto transmissionStartTime = std::chrono::high_resolution_clock::now();

            // Adjust omega for indoor/outdoor scenes - same as in serial and CUDA
            double omega = isIndoorScene ? 0.75 : 0.95;

            ImageF64 channels[3];
            split(img_double, channels, 3);

            // Normalize each channel by atmospheric light
#pragma omp parallel for
            for (int k = 0; k < 3; k++) {
                for (int i = 0; i < channels[k].height; i++) {
                    for (int j = 0; j < channels[k].width; j++) {
                        channels[k].at(i, j) /= atmospheric_light[k];
                    }
                }
            }

            // Merge normalized channels
            ImageF64 temp;
            merge(channels, 3, temp);

            // Get dark channel of normalized image
            ImageF64 temp_dark = zeros<double>(temp.width, temp.height);

#pragma omp parallel for schedule(dynamic)
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

#pragma omp parallel for collapse(2)
            for (int i = 0; i < transmission.height; i++) {
                for (int j = 0; j < transmission.width; j++) {
                    transmission.at(i, j) = 1.0 - omega * temp_dark.at(i, j);
                }
            }

            // End timing for transmission estimation
            auto transmissionEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.transmissionTime = std::chrono::duration<double, std::milli>(
                transmissionEndTime - transmissionStartTime).count();

            // Start timing for refinement
            auto refinementStartTime = std::chrono::high_resolution_clock::now();

            // Apply guided filter for refinement - use same parameters as serial and CUDA
            transmission = fastGuidedFilter(img_double, transmission, 40, 0.1, 4);

            // End timing for refinement
            auto refinementEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.refinementTime = std::chrono::duration<double, std::milli>(
                refinementEndTime - refinementStartTime).count();

            // Start timing for scene reconstruction
            auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

            // Adjust minimum transmission value based on scene type - same as serial and CUDA
            double t0 = isIndoorScene ? 0.2 : 0.1;

            // Get a fresh copy of channels
            split(img_double, channels, 3);

            // Create bounded transmission map with better sky region handling
            ImageF64 trans_bounded(transmission.width, transmission.height, 1);

#pragma omp parallel for collapse(2)
            for (int i = 0; i < trans_bounded.height; i++) {
                for (int j = 0; j < trans_bounded.width; j++) {
                    // Apply minimum transmission threshold
                    trans_bounded.at(i, j) = std::max(transmission.at(i, j), t0);

                    // Special handling for sky regions (typically in top portion of image)
                    if (i < transmission.height / 3) { // Same region as serial and CUDA
                        // Get color info to detect sky
                        double b = img_double.at(i, j, 0); // Blue
                        double g = img_double.at(i, j, 1); // Green
                        double r = img_double.at(i, j, 2); // Red

                        // Improved sky detection - same as in serial and CUDA
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

            // Process all pixels in parallel
#pragma omp parallel for collapse(2)
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

                        // Calculate luminance for saturation correction - same as in serial and CUDA
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
            timingInfo.reconstructionTime = std::chrono::duration<double, std::milli>(
                reconstructionEndTime - reconstructionStartTime).count();

            // End total time measurement
            auto totalEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.totalTime = std::chrono::duration<double, std::milli>(
                totalEndTime - totalStartTime).count();

            std::cout << "\n===== OpenMP Performance Timing (milliseconds) =====" << std::endl;
            std::cout << std::fixed << std::setprecision(10);
            std::cout << "Dark Channel Calculation: " << timingInfo.darkChannelTime << " ms" << std::endl;
            std::cout << "Atmospheric Light Estimation: " << timingInfo.atmosphericLightTime << " ms" << std::endl;
            std::cout << "Transmission Estimation: " << timingInfo.transmissionTime << " ms" << std::endl;
            std::cout << "Transmission Refinement: " << timingInfo.refinementTime << " ms" << std::endl;
            std::cout << "Scene Reconstruction: " << timingInfo.reconstructionTime << " ms" << std::endl;
            std::cout << "Total Execution Time: " << timingInfo.totalTime << " ms" << std::endl;
            std::cout << "========================================" << std::endl;

            // Save timing info in the global variable for access
            lastTimingInfo = timingInfo;

            return result_8bit;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in OpenMP implementation: " << e.what() << std::endl;
            return img;
        }
        catch (...) {
            std::cerr << "Unknown exception in dehazeParallel function" << std::endl;
            return img;
        }
    }
}