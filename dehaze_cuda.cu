// dehaze_cuda.cu
#include "dehaze.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cmath>

// Define CUDA_ENABLED for isCudaAvailable() function
#define CUDA_ENABLED

// Improved error checking macro
#define checkCudaErrors(val) { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "Error code: " << err << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
}

// Kernel error checking macro (for after kernel launches)
#define checkKernelErrors() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "Error code: " << err << std::endl; \
        throw std::runtime_error("Kernel launch error"); \
    } \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "Error code: " << err << std::endl; \
        throw std::runtime_error("Kernel execution error"); \
    } \
}

//Check CUDA Part
// Define a simple kernel
__global__ void testKernel() {
    printf("Inside CUDA kernel!\n");
}

// Define a function to launch the kernel
extern "C" void launchCudaPart() {
    try {
        printf("Launching CUDA kernel from launchCudaPart()...\n");

        // Check if device is available
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));

        if (deviceCount == 0) {
            printf("No CUDA devices found!\n");
            return;
        }

        // Get device properties
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
        printf("Using CUDA device: %s\n", deviceProp.name);

        // Launch test kernel
        testKernel << <1, 1 >> > ();
        checkKernelErrors();

        printf("Test kernel completed successfully!\n");
    }
    catch (const std::exception& e) {
        printf("CUDA test failed: %s\n", e.what());
    }
}
//Check CUDA Part End

// Optimized constant memory for performance - using double for matching precision
__constant__ double c_eps = 0.1;
__constant__ int c_window_radius = 8; // For box filter

// Improved dark channel kernel with shared memory optimization - using double precision
__global__ void darkChannelKernel(const unsigned char* imgData, double* darkChannel,
    int width, int height, int channels, int patch_radius) {
    extern __shared__ unsigned char sharedImg[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Local thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory block size with halo cells
    int block_width = blockDim.x + 2 * patch_radius;
    int block_height = blockDim.y + 2 * patch_radius;

    // Load image data into shared memory, including halo
    for (int dy = -patch_radius; dy <= patch_radius; dy += blockDim.y) {
        for (int dx = -patch_radius; dx <= patch_radius; dx += blockDim.x) {
            int sx = tx + dx + patch_radius;
            int sy = ty + dy + patch_radius;

            // Check if within shared memory bounds
            if (sx >= 0 && sx < block_width && sy >= 0 && sy < block_height) {
                int imgX = x + dx;
                int imgY = y + dy;

                // Check if within image bounds
                if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                    int shared_idx = (sy * block_width + sx) * channels;
                    int img_idx = (imgY * width + imgX) * channels;

                    // Copy RGB values
                    for (int c = 0; c < channels; c++) {
                        sharedImg[shared_idx + c] = imgData[img_idx + c];
                    }
                }
                else {
                    // Out of bounds, use a default value
                    int shared_idx = (sy * block_width + sx) * channels;
                    for (int c = 0; c < channels; c++) {
                        sharedImg[shared_idx + c] = 255; // Not dark
                    }
                }
            }
        }
    }

    __syncthreads();

    // Only process within image bounds
    if (x >= width || y >= height) return;

    // Calculate dark channel for this pixel
    double minVal = 1.0;

    for (int dy = -patch_radius; dy <= patch_radius; dy++) {
        for (int dx = -patch_radius; dx <= patch_radius; dx++) {
            int sx = tx + dx + patch_radius;
            int sy = ty + dy + patch_radius;

            // Check if within shared memory bounds
            if (sx >= 0 && sx < block_width && sy >= 0 && sy < block_height) {
                int shared_idx = (sy * block_width + sx) * channels;

                // Find minimum across RGB channels
                double b = sharedImg[shared_idx] / 255.0;
                double g = sharedImg[shared_idx + 1] / 255.0;
                double r = sharedImg[shared_idx + 2] / 255.0;

                double pixelMin = fmin(r, fmin(g, b));
                minVal = fmin(minVal, pixelMin);
            }
        }
    }

    darkChannel[y * width + x] = minVal;
}

// Convert RGB to grayscale with double precision
__global__ void rgbToGrayKernel(const unsigned char* rgb, double* gray, int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const int rgbIdx = idx * channels;

    // Convert RGB to grayscale using standard weights
    gray[idx] = (0.299 * rgb[rgbIdx + 2] + 0.587 * rgb[rgbIdx + 1] + 0.114 * rgb[rgbIdx]) / 255.0;
}

// Optimized transmission estimation kernel - double precision
__global__ void transmissionKernel(const unsigned char* imgData, const double* atmospheric,
    double* transmission, int width, int height,
    int channels, int patch_radius, double omega) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    double minVal = 1.0;
    int idx = y * width + x;

    // Search through patch around current pixel
    for (int dy = -patch_radius; dy <= patch_radius; dy++) {
        for (int dx = -patch_radius; dx <= patch_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = (ny * width + nx) * channels;

                // Normalize by atmospheric light and find minimum
                double b = imgData[nidx] / 255.0 / atmospheric[0];
                double g = imgData[nidx + 1] / 255.0 / atmospheric[1];
                double r = imgData[nidx + 2] / 255.0 / atmospheric[2];

                double pixelMin = fmin(r, fmin(g, b));
                minVal = fmin(minVal, pixelMin);
            }
        }
    }

    // Calculate transmission
    transmission[idx] = 1.0 - omega * minVal;
}

// Box filter kernel for guided filter with double precision
__global__ void boxFilterKernel(const double* input, double* output, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    double sum = 0.0;
    int count = 0;

    // Simple box filter implementation
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }
    }

    output[y * width + x] = sum / count;
}

// Element-wise multiply kernel with double precision
__global__ void multiplyKernel(const double* a, const double* b, double* result, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    result[idx] = a[idx] * b[idx];
}

// Calculate variance: var = mean_II - mean_I * mean_I
__global__ void varianceKernel(double* var_I, const double* mean_I, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    double meanI = mean_I[idx];
    var_I[idx] = var_I[idx] - meanI * meanI;
}

// Calculate covariance: cov = mean_Ip - mean_I * mean_p
__global__ void covarianceKernel(double* cov_Ip, const double* mean_I, const double* mean_p,
    double* mean_Ip, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    cov_Ip[idx] = mean_Ip[idx] - mean_I[idx] * mean_p[idx];
}

// Calculate a and b coefficients with double precision
__global__ void computeCoefficientsKernel(double* a, double* b, const double* cov_Ip,
    const double* var_I, const double* mean_I,
    const double* mean_p, int width, int height, double eps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    a[idx] = cov_Ip[idx] / (var_I[idx] + eps);
    b[idx] = mean_p[idx] - a[idx] * mean_I[idx];
}

// Calculate final guided filter result: q = a * I + b
__global__ void guidedFilterResultKernel(double* refined, const double* mean_a, const double* mean_b,
    const double* I, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    refined[idx] = mean_a[idx] * I[idx] + mean_b[idx];

    // Clamp to [0,1] range
    refined[idx] = fmax(0.0, fmin(1.0, refined[idx]));
}

// Special handling for sky regions (usually top of image)
__global__ void skyRegionHandlingKernel(double* transmission, const unsigned char* imgData,
    int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Only process top third of image (matching serial/OpenMP implementation)
    if (y < height / 3) {
        int rgbIdx = (y * width + x) * channels;
        double b = imgData[rgbIdx] / 255.0;       // Blue value
        double g = imgData[rgbIdx + 1] / 255.0;   // Green value
        double r = imgData[rgbIdx + 2] / 255.0;   // Red value

        // Match the same sky detection criteria as in serial/OpenMP implementation
        if ((b > 0.6 && b > r && b > g) || // Blue-dominant sky
            (b > 0.6 && g > 0.6 && r > 0.6)) { // Bright sky (any color)

            int idx = y * width + x;
            // Use same transmission value threshold as serial/OpenMP implementation
            transmission[idx] = fmax(transmission[idx], 0.7);
        }
    }
}

// Enhanced scene recovery with better quality and double precision
__global__ void sceneRecoveryKernel(const unsigned char* imgData, const double* transmission,
    const double* atmospheric, unsigned char* outputData,
    int width, int height, int channels, double t0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    int tidx = y * width + x;

    // Higher minimum transmission to preserve detail in dark areas
    double t = fmax(transmission[tidx], t0);

    // Create temporary storage for color values to ensure consistent processing
    double recovered[3];
    double lum = 0.0;

    // Process each channel first for recovery
    for (int c = 0; c < channels; c++) {
        double normalized = imgData[idx + c] / 255.0;
        // Apply dehaze formula J = (I-A)/t + A with bounds checking
        recovered[c] = ((normalized - atmospheric[c]) / t) + atmospheric[c];
        // Calculate lum for saturation correction (assuming BGR order)
        if (c == 0) lum += 0.114 * recovered[c]; // B
        else if (c == 1) lum += 0.587 * recovered[c]; // G
        else lum += 0.299 * recovered[c]; // R
    }

    // Now apply mild saturation correction and convert to 8-bit
    for (int c = 0; c < channels; c++) {
        // Apply correction for extreme values to reduce artifacts
        if (recovered[c] > 0.8 || recovered[c] < 0.2) {
            recovered[c] = recovered[c] * 0.85 + lum * 0.15;
        }

        // BRIGHTNESS ADJUSTMENT: Match the brightness level with serial implementation
        // Apply consistent brightness adjustments based on scene type
        if (lum < 0.5) {
            // For darker scenes (like indoor scenes), slightly increase brightness
            recovered[c] = pow(recovered[c], 0.9);
        }
        else {
            // For brighter scenes, keep as is or slightly reduce brightness
            recovered[c] = pow(recovered[c], 1.05);
        }

        // Ensure bounds and convert to 8-bit
        recovered[c] = fmin(fmax(recovered[c], 0.0), 1.0);
        outputData[idx + c] = static_cast<unsigned char>(recovered[c] * 255.0);
    }
}

// Guided filter implementation for CUDA with double precision
void guidedFilterCuda(const unsigned char* d_img, double* d_transmission, double* d_refined,
    int width, int height, int channels, int r, double eps) {
    // Allocate device memory for temporary variables
    double* d_I_gray = nullptr, * d_mean_I = nullptr, * d_mean_p = nullptr;
    double* d_mean_Ip = nullptr, * d_var_I = nullptr, * d_a = nullptr, * d_b = nullptr;
    double* d_mean_a = nullptr, * d_mean_b = nullptr, * d_cov_Ip = nullptr, * d_mean_II = nullptr;
    double* d_temp = nullptr;

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    // Allocate memory
    checkCudaErrors(cudaMalloc(&d_I_gray, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_mean_I, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_mean_p, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_mean_Ip, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_mean_II, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_var_I, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_cov_Ip, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_a, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_b, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_mean_a, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_mean_b, width * height * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_temp, width * height * sizeof(double)));

    // Convert to grayscale
    rgbToGrayKernel << <gridSize, blockSize >> > (d_img, d_I_gray, width, height, channels);
    checkKernelErrors();

    // Mean of I
    boxFilterKernel << <gridSize, blockSize >> > (d_I_gray, d_mean_I, width, height, r);
    checkKernelErrors();

    // Mean of p (transmission)
    boxFilterKernel << <gridSize, blockSize >> > (d_transmission, d_mean_p, width, height, r);
    checkKernelErrors();

    // Compute I*p
    multiplyKernel << <gridSize, blockSize >> > (d_I_gray, d_transmission, d_temp, width, height);
    checkKernelErrors();

    // Mean of I*p
    boxFilterKernel << <gridSize, blockSize >> > (d_temp, d_mean_Ip, width, height, r);
    checkKernelErrors();

    // Compute I*I
    multiplyKernel << <gridSize, blockSize >> > (d_I_gray, d_I_gray, d_temp, width, height);
    checkKernelErrors();

    // Mean of I*I
    boxFilterKernel << <gridSize, blockSize >> > (d_temp, d_mean_II, width, height, r);
    checkKernelErrors();

    // Compute variance
    varianceKernel << <gridSize, blockSize >> > (d_mean_II, d_mean_I, width, height);
    checkKernelErrors();

    // Compute covariance
    covarianceKernel << <gridSize, blockSize >> > (d_mean_Ip, d_mean_I, d_mean_p, d_mean_Ip, width, height);
    checkKernelErrors();

    // Compute a and b coefficients
    computeCoefficientsKernel << <gridSize, blockSize >> > (d_a, d_b, d_mean_Ip, d_mean_II,
        d_mean_I, d_mean_p, width, height, eps);
    checkKernelErrors();

    // Mean of a and b
    boxFilterKernel << <gridSize, blockSize >> > (d_a, d_mean_a, width, height, r);
    boxFilterKernel << <gridSize, blockSize >> > (d_b, d_mean_b, width, height, r);
    checkKernelErrors();

    // Final guided filter result
    guidedFilterResultKernel << <gridSize, blockSize >> > (d_refined, d_mean_a, d_mean_b,
        d_I_gray, width, height);
    checkKernelErrors();

    // Apply sky region handling
    skyRegionHandlingKernel << <gridSize, blockSize >> > (d_refined, d_img, width, height, channels);
    checkKernelErrors();

    // Free memory
    cudaFree(d_I_gray);
    cudaFree(d_mean_I);
    cudaFree(d_mean_p);
    cudaFree(d_mean_Ip);
    cudaFree(d_mean_II);
    cudaFree(d_var_I);
    cudaFree(d_cov_Ip);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_mean_a);
    cudaFree(d_mean_b);
    cudaFree(d_temp);
}

namespace DarkChannel {
    extern TimingInfo lastTimingInfo;

    bool isCudaAvailable() {
        try {
            int deviceCount = 0;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);
            if (error != cudaSuccess) {
                std::cerr << "CUDA device check failed: " << cudaGetErrorString(error) << std::endl;
                return false;
            }

            if (deviceCount == 0) {
                std::cerr << "No CUDA devices found on this system." << std::endl;
                return false;
            }

            // Additional check: verify device capabilities
            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

            // Check for compute capability (minimum 3.0 required)
            if (deviceProp.major < 3) {
                std::cerr << "CUDA device has compute capability "
                    << deviceProp.major << "." << deviceProp.minor
                    << ", but 3.0 or higher is required." << std::endl;
                return false;
            }

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception checking CUDA availability: " << e.what() << std::endl;
            return false;
        }
    }

    ImageU8 dehaze_cuda(const ImageU8& img) {
        try {
            // Reset timing info
            TimingInfo timingInfo;

            // Start total time measurement
            auto totalStartTime = std::chrono::high_resolution_clock::now();

            if (!isCudaAvailable()) {
                std::cerr << "No suitable CUDA device found. Falling back to CPU implementation." << std::endl;
                return dehaze(img);
            }

            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
            std::cout << "Using CUDA device: " << deviceProp.name << " with compute capability "
                << deviceProp.major << "." << deviceProp.minor << std::endl;

            // Check if image is valid
            if (img.empty() || img.channels != 3) {
                std::cerr << "Error: Input image is empty or not 3-channel" << std::endl;
                return img;
            }

            // Image dimensions
            int width = img.width;
            int height = img.height;
            int channels = img.channels;
            int imgSize = width * height;
            size_t imgBytes = imgSize * channels * sizeof(unsigned char);

            std::cout << "Processing image size: " << width << "x" << height << std::endl;

            // Allocate host and device memory
            unsigned char* d_img = nullptr;
            double* d_darkChannel = nullptr;
            double* d_transmission = nullptr;
            double* d_refined_transmission = nullptr;
            unsigned char* d_result = nullptr;
            double* d_atmospheric = nullptr;

            try {
                // Start dark channel calculation timing
                auto darkChannelStartTime = std::chrono::high_resolution_clock::now();

                // Allocate device memory
                checkCudaErrors(cudaMalloc(&d_img, imgBytes));
                checkCudaErrors(cudaMalloc(&d_darkChannel, imgSize * sizeof(double)));

                // Copy input image to device
                checkCudaErrors(cudaMemcpy(d_img, img.get_data(), imgBytes, cudaMemcpyHostToDevice));

                // Define block and grid dimensions - adjust for large images
                dim3 blockSize(16, 16);
                dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                    (height + blockSize.y - 1) / blockSize.y);

                // Calculate shared memory size for the dark channel kernel
                int patchRadius = 7;
                int sharedWidth = blockSize.x + 2 * patchRadius;
                int sharedHeight = blockSize.y + 2 * patchRadius;
                size_t sharedMemSize = sharedWidth * sharedHeight * channels * sizeof(unsigned char);

                // Dark channel calculation with shared memory
                darkChannelKernel << <gridSize, blockSize, sharedMemSize >> > (
                    d_img, d_darkChannel, width, height, channels, patchRadius);
                checkKernelErrors();

                // Copy dark channel back to host for atmospheric light estimation
                double* h_darkChannel = new double[imgSize];
                checkCudaErrors(cudaMemcpy(h_darkChannel, d_darkChannel,
                    imgSize * sizeof(double), cudaMemcpyDeviceToHost));

                // End dark channel timing
                auto darkChannelEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(
                    darkChannelEndTime - darkChannelStartTime).count();

                // Start atmospheric light estimation timing
                auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

                // Find atmospheric light (top 0.1% brightest pixels in dark channel)
                std::vector<std::pair<double, int>> brightnessIndices;
                brightnessIndices.reserve(imgSize);

                for (int i = 0; i < imgSize; i++) {
                    brightnessIndices.push_back(std::make_pair(h_darkChannel[i], i));
                }

                std::sort(brightnessIndices.begin(), brightnessIndices.end(),
                    [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                        return a.first > b.first;
                    });

                // Use top 0.1% brightest pixels but with protection for extreme values
                int numBrightestPixels = std::max(1, imgSize / 1000);
                double atmospheric[3] = { 0, 0, 0 };
                double maxAtmospheric = 0.0;
                bool validAtmospheric = false;

                // First pass - find the maximum value
                for (int i = 0; i < numBrightestPixels; i++) {
                    int idx = brightnessIndices[i].second;
                    int y = idx / width;
                    int x = idx % width;

                    double avgIntensity = (img.at(y, x, 0) + img.at(y, x, 1) + img.at(y, x, 2)) / 3.0;

                    if (avgIntensity > maxAtmospheric) {
                        maxAtmospheric = avgIntensity;
                    }
                }

                // Now use pixels that are not too far from the max
                for (int i = 0; i < numBrightestPixels; i++) {
                    int idx = brightnessIndices[i].second;
                    int y = idx / width;
                    int x = idx % width;

                    double avgIntensity = (img.at(y, x, 0) + img.at(y, x, 1) + img.at(y, x, 2)) / 3.0;

                    // Only use pixels that are reasonably bright
                    if (avgIntensity > maxAtmospheric * 0.7) {
                        atmospheric[0] += img.at(y, x, 0) / 255.0;  // B
                        atmospheric[1] += img.at(y, x, 1) / 255.0;  // G
                        atmospheric[2] += img.at(y, x, 2) / 255.0;  // R
                        validAtmospheric = true;
                    }
                }

                // Normalize if we found valid pixels
                if (validAtmospheric) {
                    atmospheric[0] /= numBrightestPixels;
                    atmospheric[1] /= numBrightestPixels;
                    atmospheric[2] /= numBrightestPixels;
                }
                else {
                    // Use a fallback if no good atmospheric light found
                    atmospheric[0] = 0.8;
                    atmospheric[1] = 0.8;
                    atmospheric[2] = 0.8;
                }

                // Use exactly the same bounds in both serial and CUDA
                for (int i = 0; i < 3; i++) {
                    atmospheric[i] = std::max(0.05, std::min(0.95, atmospheric[i]));
                }

                // Check if this is likely an indoor scene based on atmospheric light
                bool isIndoorScene = false;
                double avgAtmospheric = (atmospheric[0] + atmospheric[1] + atmospheric[2]) / 3.0;
                if (avgAtmospheric < 0.6) {
                    isIndoorScene = true;
                }

                // End atmospheric light timing
                auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(
                    atmosphericEndTime - atmosphericStartTime).count();

                // Start transmission estimation timing
                auto transmissionStartTime = std::chrono::high_resolution_clock::now();

                // Allocate memory for transmission map
                checkCudaErrors(cudaMalloc(&d_transmission, imgSize * sizeof(double)));

                // Copy atmospheric light to device
                checkCudaErrors(cudaMalloc(&d_atmospheric, 3 * sizeof(double)));
                checkCudaErrors(cudaMemcpy(d_atmospheric, atmospheric,
                    3 * sizeof(double), cudaMemcpyHostToDevice));

                // Adjust omega for indoor scenes - same as serial
                double omega = isIndoorScene ? 0.75 : 0.95;

                // Estimate transmission using dark channel prior
                transmissionKernel << <gridSize, blockSize >> > (
                    d_img, d_atmospheric, d_transmission, width, height, channels, patchRadius, omega);
                checkKernelErrors();

                // End transmission timing
                auto transmissionEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.transmissionTime = std::chrono::duration<double, std::milli>(
                    transmissionEndTime - transmissionStartTime).count();

                // Start refinement timing
                auto refinementStartTime = std::chrono::high_resolution_clock::now();

                // Allocate memory for refined transmission
                checkCudaErrors(cudaMalloc(&d_refined_transmission, imgSize * sizeof(double)));

                // Apply guided filter refinement using double precision implementation
                // Use the same parameters as serial implementation (radius 40, epsilon 0.1, subsample factor 4)
                guidedFilterCuda(d_img, d_transmission, d_refined_transmission, width, height, channels, 40, 0.1);

                // End refinement timing
                auto refinementEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.refinementTime = std::chrono::duration<double, std::milli>(
                    refinementEndTime - refinementStartTime).count();

                // Start scene reconstruction timing
                auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

                // Allocate memory for result
                checkCudaErrors(cudaMalloc(&d_result, imgBytes));

                // Adjust minimum transmission value for indoor scenes - same as serial
                double t0 = isIndoorScene ? 0.2 : 0.1;

                // Recover scene with enhanced quality and adaptive parameters
                sceneRecoveryKernel << <gridSize, blockSize >> > (
                    d_img, d_refined_transmission, d_atmospheric, d_result, width, height, channels, t0);
                checkKernelErrors();

                // Copy result back to host
                ImageU8 result(width, height, channels);
                checkCudaErrors(cudaMemcpy(result.get_data(), d_result,
                    imgBytes, cudaMemcpyDeviceToHost));

                // End scene reconstruction timing
                auto reconstructionEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.reconstructionTime = std::chrono::duration<double, std::milli>(
                    reconstructionEndTime - reconstructionStartTime).count();

                // End total time measurement
                auto totalEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.totalTime = std::chrono::duration<double, std::milli>(
                    totalEndTime - totalStartTime).count();

                std::cout << "\n===== CUDA Performance Timing (milliseconds) =====" << std::endl;
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

                // Cleanup
                delete[] h_darkChannel;
                cudaFree(d_img);
                cudaFree(d_darkChannel);
                cudaFree(d_transmission);
                cudaFree(d_refined_transmission);
                cudaFree(d_atmospheric);
                cudaFree(d_result);

                return result;
            }
            catch (const std::exception& e) {
                std::cerr << "Exception during CUDA processing: " << e.what() << std::endl;

                // Cleanup - free allocated memory to avoid memory leaks
                if (d_img) cudaFree(d_img);
                if (d_darkChannel) cudaFree(d_darkChannel);
                if (d_transmission) cudaFree(d_transmission);
                if (d_refined_transmission) cudaFree(d_refined_transmission);
                if (d_atmospheric) cudaFree(d_atmospheric);
                if (d_result) cudaFree(d_result);

                std::cerr << "Falling back to CPU implementation." << std::endl;
                return dehaze(img);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in CUDA implementation: " << e.what() << std::endl;
            return img;
        }
        catch (...) {
            std::cerr << "Unknown exception in dehaze_cuda function" << std::endl;
            return img;
        }
    }
}