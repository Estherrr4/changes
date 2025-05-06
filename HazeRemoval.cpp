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

    cv::Mat dehaze(const cv::Mat& img) {
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
            __Assert__(img.type() == CV_8UC3, "3 channel images only.");

            // Convert to double precision for calculations
            cv::Mat img_double;
            img.convertTo(img_double, CV_64FC3);
            img_double /= 255;

            // Start timing for dark channel calculation
            auto darkChannelStartTime = std::chrono::high_resolution_clock::now();

            // Get dark channel
            int patch_radius = 7;
            cv::Mat darkchannel = cv::Mat::zeros(img_double.rows, img_double.cols, CV_64F);
            for (int i = 0; i < darkchannel.rows; i++) {
                for (int j = 0; j < darkchannel.cols; j++) {
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(darkchannel.rows, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(darkchannel.cols, j + patch_radius + 1);
                    double dark = 1.0;  // Initialize to max normalized value (1.0)
                    for (int r = r_start; r < r_end; r++) {
                        for (int c = c_start; c < c_end; c++) {
                            const cv::Vec3d& val = img_double.at<cv::Vec3d>(r, c);
                            dark = std::min(dark, val[0]);
                            dark = std::min(dark, val[1]);
                            dark = std::min(dark, val[2]);
                        }
                    }
                    darkchannel.at<double>(i, j) = dark;
                }
            }

            // End timing for dark channel calculation
            auto darkChannelEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(darkChannelEndTime - darkChannelStartTime).count();

            // Start timing for atmospheric light estimation
            auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

            // Get atmospheric light
            int pixels = img_double.rows * img_double.cols;
            // Ensure at least 1 pixel is used
            int num = std::max(1, pixels / 1000); // 0.1% brightest pixels
            std::vector<std::pair<double, int>> V;
            V.reserve(pixels);

            int k = 0;
            for (int i = 0; i < darkchannel.rows; i++) {
                for (int j = 0; j < darkchannel.cols; j++) {
                    V.emplace_back(darkchannel.at<double>(i, j), k);
                    k++;
                }
            }

            std::sort(V.begin(), V.end(), [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.first > p2.first;
                });

            double atmospheric_light[3] = { 0, 0, 0 };
            double maxAtmospheric = 0.0;
            for (k = 0; k < num; k++) {
                int r = V[k].second / darkchannel.cols;
                int c = V[k].second % darkchannel.cols;
                const cv::Vec3d& val = img_double.at<cv::Vec3d>(r, c);

                double avgIntensity = (val[0] + val[1] + val[2]) / 3.0;
                if (avgIntensity > maxAtmospheric) {
                    maxAtmospheric = avgIntensity;
                }
            }

            // Second pass - use only reasonably bright pixels
            int validPixels = 0;
            for (k = 0; k < num; k++) {
                int r = V[k].second / darkchannel.cols;
                int c = V[k].second % darkchannel.cols;
                const cv::Vec3d& val = img_double.at<cv::Vec3d>(r, c);

                double avgIntensity = (val[0] + val[1] + val[2]) / 3.0;
                if (avgIntensity > maxAtmospheric * 0.7) {
                    atmospheric_light[0] += val[0];
                    atmospheric_light[1] += val[1];
                    atmospheric_light[2] += val[2];
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

            // Check if this is likely an indoor scene
            bool isIndoorScene = false;
            double avgAtmospheric = (atmospheric_light[0] + atmospheric_light[1] + atmospheric_light[2]) / 3.0;
            if (avgAtmospheric < 0.6) {
                isIndoorScene = true;
            }

            // End timing for atmospheric light estimation
            auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(atmosphericEndTime - atmosphericStartTime).count();

            // Start timing for transmission estimation
            auto transmissionStartTime = std::chrono::high_resolution_clock::now();

            // Adjust omega for indoor/outdoor scenes
            double omega = isIndoorScene ? 0.75 : 0.95;
            cv::Mat channels[3];
            cv::split(img_double, channels);
            for (k = 0; k < 3; k++) {
                channels[k] /= atmospheric_light[k];
            }

            cv::Mat temp;
            cv::merge(channels, 3, temp);

            // Get dark channel of normalized image
            cv::Mat temp_dark = cv::Mat::zeros(temp.rows, temp.cols, CV_64F);
            for (int i = 0; i < temp.rows; i++) {
                for (int j = 0; j < temp.cols; j++) {
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(temp.rows, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(temp.cols, j + patch_radius + 1);
                    double dark = 1.0;
                    for (int r = r_start; r < r_end; r++) {
                        for (int c = c_start; c < c_end; c++) {
                            const cv::Vec3d& val = temp.at<cv::Vec3d>(r, c);
                            dark = std::min(dark, val[0]);
                            dark = std::min(dark, val[1]);
                            dark = std::min(dark, val[2]);
                        }
                    }
                    temp_dark.at<double>(i, j) = dark;
                }
            }

            // Get transmission
            cv::Mat transmission = 1.0 - omega * temp_dark;

            // End timing for transmission estimation
            auto transmissionEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.transmissionTime = std::chrono::duration<double, std::milli>(transmissionEndTime - transmissionStartTime).count();

            // Start timing for refinement
            auto refinementStartTime = std::chrono::high_resolution_clock::now();

            // Apply guided filter for refinement - use consistent parameters
            transmission = fastGuidedFilter(img_double, transmission, 40, 0.1, 4);

            // End timing for refinement
            auto refinementEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.refinementTime = std::chrono::duration<double, std::milli>(refinementEndTime - refinementStartTime).count();

            // Start timing for scene reconstruction
            auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

            // Adjust minimum transmission value based on scene type
            double t0 = isIndoorScene ? 0.2 : 0.1;

            // Get a fresh copy of channels
            cv::split(img_double, channels);

            // Create bounded transmission map with better sky region handling
            cv::Mat trans_ = cv::Mat(transmission.size(), CV_64F);
            for (int i = 0; i < transmission.rows; i++) {
                for (int j = 0; j < transmission.cols; j++) {
                    // Apply minimum transmission threshold
                    trans_.at<double>(i, j) = std::max(transmission.at<double>(i, j), t0);

                    // Special handling for sky regions (typically in top portion of image)
                    if (i < transmission.rows / 2) {
                        // Get color info to detect sky
                        const cv::Vec3d& val = img_double.at<cv::Vec3d>(i, j);
                        double b = val[0]; // Blue
                        double g = val[1]; // Green
                        double r = val[2]; // Red

                        // Improved sky detection
                        if (b > 0.5 || (b > 0.4 && g > 0.4 && r > 0.4)) {
                            // Use higher transmission for sky
                            trans_.at<double>(i, j) = 0.95;

                            // Special handling for horizon/upper sky
                            if (i < transmission.rows / 6) {
                                trans_.at<double>(i, j) = 0.99;
                            }
                        }
                    }

                    // Special handling for very dark regions
                    if (darkchannel.at<double>(i, j) < 0.02) {
                        trans_.at<double>(i, j) = std::max(trans_.at<double>(i, j), 0.4);
                    }
                }
            }

            // Process each channel with improved scene reconstruction
            cv::Mat result_channels[3];
            for (int c = 0; c < 3; c++) {
                result_channels[c] = cv::Mat(img_double.size().height, img_double.size().width, CV_64F);
            }

            // Recover the scene
            for (int i = 0; i < img_double.rows; i++) {
                for (int j = 0; j < img_double.cols; j++) {
                    double t = trans_.at<double>(i, j);

                    // Process each channel with the dehaze formula
                    for (int c = 0; c < 3; c++) {
                        double val = img_double.at<cv::Vec3d>(i, j)[c];
                        double A = atmospheric_light[c];

                        // Apply recovery formula with improved stability
                        double recovered = ((val - A) / t) + A;
                        recovered = std::max(0.0, std::min(1.0, recovered));

                        result_channels[c].at<double>(i, j) = recovered;
                    }
                }
            }

            // Merge channels
            cv::Mat res;
            cv::merge(result_channels, 3, res);

            // Apply final color adjustment and tone mapping
            for (int i = 0; i < res.rows; i++) {
                for (int j = 0; j < res.cols; j++) {
                    cv::Vec3d& val = res.at<cv::Vec3d>(i, j);

                    // Calculate luminance for adjustment
                    double lum = 0.299 * val[2] + 0.587 * val[1] + 0.114 * val[0];

                    // Apply mild saturation correction for extreme values
                    for (int c = 0; c < 3; c++) {
                        if (val[c] > 0.8 || val[c] < 0.2) {
                            val[c] = val[c] * 0.85 + lum * 0.15;
                        }

                        // Apply brightness adjustment based on scene type
                        if (lum < 0.5) {
                            // Brighten darker scenes
                            val[c] = std::pow(val[c], 0.9);
                        }
                        else {
                            // Slightly reduce brightness in bright scenes
                            val[c] = std::pow(val[c], 1.05);
                        }

                        // Final bounds check
                        val[c] = std::max(0.0, std::min(1.0, val[c]));
                    }
                }
            }

            // Convert back to 8-bit format
            res *= 255.0;
            cv::Mat result;
            res.convertTo(result, CV_8UC3);

            // End timing for scene reconstruction
            auto reconstructionEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.reconstructionTime = std::chrono::duration<double, std::milli>(reconstructionEndTime - reconstructionStartTime).count();

            // End total time measurement
            auto totalEndTime = std::chrono::high_resolution_clock::now();
            lastTimingInfo.totalTime = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();

            std::cout << "\n===== Performance Timing (milliseconds) =====" << std::endl;
            std::cout << std::fixed << std::setprecision(10);
            std::cout << "Dark Channel Calculation: " << lastTimingInfo.darkChannelTime << " ms" << std::endl;
            std::cout << "Atmospheric Light Estimation: " << lastTimingInfo.atmosphericLightTime << " ms" << std::endl;
            std::cout << "Transmission Estimation: " << lastTimingInfo.transmissionTime << " ms" << std::endl;
            std::cout << "Transmission Refinement: " << lastTimingInfo.refinementTime << " ms" << std::endl;
            std::cout << "Scene Reconstruction: " << lastTimingInfo.reconstructionTime << " ms" << std::endl;
            std::cout << "Total Execution Time: " << lastTimingInfo.totalTime << " ms" << std::endl;
            std::cout << "========================================" << std::endl;

            return result;
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception: " << e.what() << std::endl;
            return img;
        }
        catch (const std::exception& e) {
            std::cerr << "Standard Exception: " << e.what() << std::endl;
            return img;
        }
        catch (...) {
            std::cerr << "Unknown exception in dehaze function" << std::endl;
            return img;
        }
    }
}