#include "dehaze.h"
#include "dehaze_parallel.h"
#include "custom_types.h"
#include "custom_image_utils.h" // Include our new custom image utils
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <omp.h>

#ifdef _WIN32
#include <Windows.h>
#include <commdlg.h>
#include <conio.h> // For _getch() on Windows
#endif

// Declare CUDA function at global scope - THIS IS KEY TO FIX THE LINKAGE ERROR
extern "C" void launchCudaPart();

inline int get_max(int a, int b) {
    return (a > b) ? a : b;
}

// Simple function to check if a file exists
bool fileExists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

// Function to wait for a key press without blocking the return to menu
void waitForKeyPress() {
    std::cout << "\nPress any key to return to menu..." << std::endl;

#ifdef _WIN32
    _getch(); // Use _getch() on Windows which doesn't require Enter key
#else
    // For non-Windows platforms
    getchar();
#endif
}

// Function to show file open dialog (ANSI version)
std::string getImageFile() {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    char fileName[MAX_PATH] = "";
    ZeroMemory(&ofn, sizeof(ofn));

    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Image Files (*.jpg;*.jpeg;*.png;*.bmp)\0*.jpg;*.jpeg;*.png;*.bmp\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrDefExt = "jpg";

    if (GetOpenFileNameA(&ofn)) {
        return std::string(fileName);
    }
    return "";
#else
    // For non-Windows platforms, use simple input
    std::string filePath;
    std::cout << "Enter path to image file: ";
    std::getline(std::cin, filePath);
    return filePath;
#endif
}

// Function declarations for processImage and exportPerformanceDataToCSV
//bool processImage(const std::string& img_path, const std::string& project_path);
// Add this implementation in main.cpp after the function declarations
// and before the main function
bool processImage(const std::string& img_path, const std::string& project_path) {
    try {
        // Use forward slashes consistently
        std::string normalized_path = img_path;
        std::replace(normalized_path.begin(), normalized_path.end(), '\\', '/');

        std::cout << "Processing image: " << normalized_path << std::endl;

        // Load image using custom loader instead of OpenCV
        ImageU8 original_img;

        // Check file extension
        std::string ext = normalized_path.substr(normalized_path.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == "bmp") {
            // Load BMP directly with our custom loader
            original_img = ImageUtils::loadBMP(normalized_path);
        }
        else {
            // For other formats, temporarily use OpenCV just for loading
            // This could be replaced with custom loaders for other formats
            cv::Mat cv_original_img = cv::imread(normalized_path);
            if (cv_original_img.empty()) {
                std::cerr << "Error: Failed to load image: " << normalized_path << std::endl;
                waitForKeyPress();
                return false;
            }
            original_img = DarkChannel::matToImage(cv_original_img);
        }

        if (original_img.empty()) {
            std::cerr << "Error: Failed to load image: " << normalized_path << std::endl;
            waitForKeyPress();
            return false;
        }

        // Extract just the filename for the output file
        size_t pos = normalized_path.find_last_of('/');
        std::string img_name = (pos != std::string::npos) ? normalized_path.substr(pos + 1) : normalized_path;

        std::cout << "Image loaded successfully. Original size: " << original_img.width << "x" << original_img.height << std::endl;

        // Resize large images for faster processing - USING OUR CUSTOM RESIZE
        ImageU8 img;
        int maxDimension = 1200;
        if (original_img.width > maxDimension || original_img.height > maxDimension) {
            double scale = static_cast<double>(maxDimension) /
                (original_img.width > original_img.height ? original_img.width : original_img.height);
            int newWidth = static_cast<int>(original_img.width * scale);
            int newHeight = static_cast<int>(original_img.height * scale);

            // Use our custom resize instead of OpenCV
            img = ImageUtils::customResize(original_img, newWidth, newHeight);
            std::cout << "Image resized from " << original_img.width << "x" << original_img.height
                << " to " << img.width << "x" << img.height << std::endl;
        }
        else {
            img = original_img;
        }

        // Save original and resized version of input image for reference using custom BMP saver
        std::string original_path = project_path + "original_" + img_name.substr(0, img_name.find_last_of(".")) + ".bmp";
        ImageUtils::saveBMP(original_img, original_path);

        std::string resized_input_path;
        if (img.get_data() != original_img.get_data()) {
            resized_input_path = project_path + "resized_input_" + img_name.substr(0, img_name.find_last_of(".")) + ".bmp";
            ImageUtils::saveBMP(img, resized_input_path);
            std::cout << "Original image saved as: " << original_path << std::endl;
            std::cout << "Resized input saved as: " << resized_input_path << std::endl;
        }
        else {
            resized_input_path = original_path;
        }

        // Process with serial version
        std::cout << "\nRunning serial version..." << std::endl;
        ImageU8 serial_result = DarkChannel::dehaze(img);
        DarkChannel::TimingInfo serial_timing = DarkChannel::getLastTimingInfo();

        // Process with OpenMP version
        std::cout << "\nRunning OpenMP version..." << std::endl;
        int num_threads = get_max(1, omp_get_max_threads());
        std::cout << "Using " << num_threads << " OpenMP threads" << std::endl;
        ImageU8 openmp_result = DarkChannel::dehazeParallel(img, num_threads);
        DarkChannel::TimingInfo openmp_timing = DarkChannel::getLastTimingInfo();

        // Check CUDA availability
        bool cuda_available = DarkChannel::isCudaAvailable();
        ImageU8 cuda_result;
        DarkChannel::TimingInfo cuda_timing;

        if (cuda_available) {
            // Run CUDA implementation if available
            std::cout << "\nRunning CUDA version..." << std::endl;
            try {
                cuda_result = DarkChannel::dehaze_cuda(img);
                cuda_timing = DarkChannel::getLastTimingInfo();

                // Verify the CUDA result is valid
                if (cuda_result.empty()) {
                    std::cerr << "Warning: CUDA result is empty" << std::endl;
                    cuda_available = false;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA processing failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation." << std::endl;
                cuda_available = false;
            }
        }
        else {
            std::cout << "\nCUDA implementation not available." << std::endl;
        }

        // Print timing comparison table
        std::cout << "\n===== DEHAZING PERFORMANCE COMPARISON (milliseconds) =====" << std::endl;
        std::cout << std::left << std::setw(25) << "\nStage";
        std::cout << std::right << std::setw(17) << "Serial";
        std::cout << std::setw(17) << "OpenMP";
        if (cuda_available) std::cout << std::setw(17) << "CUDA";
        std::cout << std::setw(17) << "OMP Speedup";
        if (cuda_available) std::cout << std::setw(17) << "CUDA Speedup";
        std::cout << std::endl;

        int dividerWidth = 25 + 17 + 17 + 17 + (cuda_available ? 17 + 19 : 0);
        std::cout << std::string(dividerWidth, '-') << std::endl;

        std::cout << std::fixed << std::setprecision(10);

        std::cout << std::left << std::setw(25) << "Dark Channel Calculation";
        std::cout << std::right << std::setw(17) << serial_timing.darkChannelTime;
        std::cout << std::setw(17) << openmp_timing.darkChannelTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.darkChannelTime;
        }

        double omp_speedup = (serial_timing.darkChannelTime > 0) ?
            (serial_timing.darkChannelTime / openmp_timing.darkChannelTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.darkChannelTime > 0) ?
                (serial_timing.darkChannelTime / cuda_timing.darkChannelTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        // Output remaining timing stages
        std::cout << std::left << std::setw(25) << "Atmospheric Light";
        std::cout << std::right << std::setw(17) << serial_timing.atmosphericLightTime;
        std::cout << std::setw(17) << openmp_timing.atmosphericLightTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.atmosphericLightTime;
        }

        omp_speedup = (serial_timing.atmosphericLightTime > 0) ?
            (serial_timing.atmosphericLightTime / openmp_timing.atmosphericLightTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.atmosphericLightTime > 0) ?
                (serial_timing.atmosphericLightTime / cuda_timing.atmosphericLightTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(25) << "Transmission Estimation";
        std::cout << std::right << std::setw(17) << serial_timing.transmissionTime;
        std::cout << std::setw(17) << openmp_timing.transmissionTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.transmissionTime;
        }

        omp_speedup = (serial_timing.transmissionTime > 0) ?
            (serial_timing.transmissionTime / openmp_timing.transmissionTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.transmissionTime > 0) ?
                (serial_timing.transmissionTime / cuda_timing.transmissionTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(25) << "Transmission Refinement";
        std::cout << std::right << std::setw(17) << serial_timing.refinementTime;
        std::cout << std::setw(17) << openmp_timing.refinementTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.refinementTime;
        }

        omp_speedup = (serial_timing.refinementTime > 0) ?
            (serial_timing.refinementTime / openmp_timing.refinementTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.refinementTime > 0) ?
                (serial_timing.refinementTime / cuda_timing.refinementTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(25) << "Scene Reconstruction";
        std::cout << std::right << std::setw(17) << serial_timing.reconstructionTime;
        std::cout << std::setw(17) << openmp_timing.reconstructionTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.reconstructionTime;
        }

        omp_speedup = (serial_timing.reconstructionTime > 0) ?
            (serial_timing.reconstructionTime / openmp_timing.reconstructionTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.reconstructionTime > 0) ?
                (serial_timing.reconstructionTime / cuda_timing.reconstructionTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(25) << "Total Execution Time";
        std::cout << std::right << std::setw(17) << serial_timing.totalTime;
        std::cout << std::setw(17) << openmp_timing.totalTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.totalTime;
        }

        omp_speedup = (serial_timing.totalTime > 0) ?
            (serial_timing.totalTime / openmp_timing.totalTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.totalTime > 0) ?
                (serial_timing.totalTime / cuda_timing.totalTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::string(dividerWidth, '=') << std::endl;

        // Save results using custom BMP saver instead of OpenCV
        std::string serial_output_path = project_path + "serial_dehazed_" + img_name.substr(0, img_name.find_last_of(".")) + ".bmp";
        std::string openmp_output_path = project_path + "openmp_dehazed_" + img_name.substr(0, img_name.find_last_of(".")) + ".bmp";
        std::string cuda_output_path;

        ImageUtils::saveBMP(serial_result, serial_output_path);
        ImageUtils::saveBMP(openmp_result, openmp_output_path);

        if (cuda_available) {
            cuda_output_path = project_path + "cuda_dehazed_" + img_name.substr(0, img_name.find_last_of(".")) + ".bmp";
            ImageUtils::saveBMP(cuda_result, cuda_output_path);
        }

        std::cout << "Results saved as: " << std::endl;
        std::cout << "  Serial: " << serial_output_path << std::endl;
        std::cout << "  OpenMP: " << openmp_output_path << std::endl;
        if (cuda_available) {
            std::cout << "  CUDA: " << cuda_output_path << std::endl;
        }

        // Display results using our custom HTML viewer instead of OpenCV
        std::cout << "\nPreparing to display results..." << std::endl;

        if (cuda_available) {
            ImageUtils::displayResults(
                project_path,
                img_name,
                original_path,
                serial_output_path,
                openmp_output_path,
                cuda_output_path
            );
        }
        else {
            ImageUtils::displayResults(
                project_path,
                img_name,
                original_path,
                serial_output_path,
                openmp_output_path
            );
        }

        waitForKeyPress();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in processImage: " << e.what() << std::endl;
        waitForKeyPress();
        return false;
    }
}

void exportPerformanceDataToCSV(
    const std::string& filename,
    const std::vector<std::string>& imagePaths,
    const std::vector<std::pair<DarkChannel::TimingInfo, DarkChannel::TimingInfo>>& timingPairs,
    bool cudaAvailable,
    const std::vector<DarkChannel::TimingInfo>* cudaTimings) {

    std::ofstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    csvFile << "Image,ImageSize,Orientation,Stage,Serial_Time,OpenMP_Time,OpenMP_Speedup";
    if (cudaAvailable) {
        csvFile << ",CUDA_Time,CUDA_Speedup";
    }
    csvFile << std::endl;

    for (size_t i = 0; i < imagePaths.size(); i++) {
        // Use our custom function to get image dimensions instead of OpenCV
        auto dimensions = ImageUtils::getImageDimensions(imagePaths[i]);
        int width = dimensions.first;
        int height = dimensions.second;

        std::string orientation = (height > width) ? "Portrait" : "Landscape";
        std::string imageSize = std::to_string(width) + "x" + std::to_string(height);

        // Extract just the filename without path
        size_t pos = imagePaths[i].find_last_of("/\\");
        std::string imageName = (pos != std::string::npos) ?
            imagePaths[i].substr(pos + 1) : imagePaths[i];

        const DarkChannel::TimingInfo& serial = timingPairs[i].first;
        const DarkChannel::TimingInfo& openmp = timingPairs[i].second;

        auto writeRow = [&](const std::string& stage, double serialTime, double openmpTime) {
            double openmpSpeedup = (serialTime > 0) ? (serialTime / openmpTime) : 0.0;

            csvFile << imageName << "," << imageSize << "," << orientation << ","
                << stage << "," << serialTime << "," << openmpTime << ","
                << openmpSpeedup;

            if (cudaAvailable && cudaTimings) {
                double cudaTime = 0.0;

                if (stage == "Dark Channel") {
                    cudaTime = (*cudaTimings)[i].darkChannelTime;
                }
                else if (stage == "Atmospheric Light") {
                    cudaTime = (*cudaTimings)[i].atmosphericLightTime;
                }
                else if (stage == "Transmission") {
                    cudaTime = (*cudaTimings)[i].transmissionTime;
                }
                else if (stage == "Refinement") {
                    cudaTime = (*cudaTimings)[i].refinementTime;
                }
                else if (stage == "Reconstruction") {
                    cudaTime = (*cudaTimings)[i].reconstructionTime;
                }
                else if (stage == "Total") {
                    cudaTime = (*cudaTimings)[i].totalTime;
                }

                double cudaSpeedup = (serialTime > 0) ? (serialTime / cudaTime) : 0.0;
                csvFile << "," << cudaTime << "," << cudaSpeedup;
            }

            csvFile << std::endl;
            };

        writeRow("Dark Channel", serial.darkChannelTime, openmp.darkChannelTime);
        writeRow("Atmospheric Light", serial.atmosphericLightTime, openmp.atmosphericLightTime);
        writeRow("Transmission", serial.transmissionTime, openmp.transmissionTime);
        writeRow("Refinement", serial.refinementTime, openmp.refinementTime);
        writeRow("Reconstruction", serial.reconstructionTime, openmp.reconstructionTime);
        writeRow("Total", serial.totalTime, openmp.totalTime);
    }

    csvFile.close();
    std::cout << "Performance data exported to " << filename << std::endl;
}



// Main function
int main(int argc, char** argv) {
    try {
        // Test CUDA functionality
        std::cout << "Testing CUDA functionality..." << std::endl;
        try {
            // Call the CUDA test function (declared at global scope)
            launchCudaPart();
            std::cout << "CUDA test completed." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "CUDA test failed: " << e.what() << std::endl;
            std::cerr << "Some features may be unavailable." << std::endl;
        }

        std::string project_path = "./";

        bool interactive_mode = true;
        std::string img_path;

        if (argc > 1) {
            interactive_mode = false;
            std::string arg_path = argv[1];

            if (arg_path.find(':') == std::string::npos) {
                img_path = project_path + arg_path;
            }
            else {
                img_path = arg_path;
            }

            processImage(img_path, project_path);

            interactive_mode = true;
        }

        if (interactive_mode) {
            int choice = -1;
            while (choice != 0) {
                std::cout << "\n=== Dehaze Interactive Mode ===" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  1: Enter image path manually" << std::endl;
                std::cout << "  2: Use file dialog to select image" << std::endl;
                std::cout << "  3: Process multiple images" << std::endl;
                std::cout << "  4: Check CUDA availability" << std::endl;
                std::cout << "  5: Run performance benchmark (for plotting)" << std::endl;
                std::cout << "  0: Exit" << std::endl;

                std::cout << "\nEnter your choice (0-5): ";
                std::cin >> choice;
                std::cin.ignore();

                switch (choice) {
                case 0:
                    std::cout << "Exiting program." << std::endl;
                    break;

                case 1: {
                    // Manual path entry
                    std::cout << "Enter image path (or 'exit' to return to menu): ";
                    std::getline(std::cin, img_path);
                    if (img_path == "exit" || img_path == "quit") {
                        break;
                    }
                    processImage(img_path, project_path);
                    break;
                }

                case 2: {
                    // File dialog
                    img_path = getImageFile();
                    if (!img_path.empty()) {
                        processImage(img_path, project_path);
                    }
                    else {
                        std::cout << "No file selected." << std::endl;
                        waitForKeyPress();
                    }
                    break;
                }

                case 3: {
                    // Process multiple images
                    std::cout << "Enter 'exit' at any time to return to menu." << std::endl;
                    while (true) {
                        std::cout << "\nSelect option for next image:" << std::endl;
                        std::cout << "  1: Enter path manually" << std::endl;
                        std::cout << "  2: Use file dialog" << std::endl;
                        std::cout << "  0: Return to main menu" << std::endl;

                        int subChoice;
                        std::cout << "Choice: ";
                        std::cin >> subChoice;
                        std::cin.ignore();

                        if (subChoice == 0) {
                            break;
                        }
                        else if (subChoice == 1) {
                            std::cout << "Enter image path: ";
                            std::getline(std::cin, img_path);
                            if (img_path == "exit" || img_path == "quit") {
                                break;
                            }
                            processImage(img_path, project_path);
                        }
                        else if (subChoice == 2) {
                            img_path = getImageFile();
                            if (img_path.empty()) {
                                std::cout << "No file selected." << std::endl;
                                waitForKeyPress();
                                continue;
                            }
                            processImage(img_path, project_path);
                        }
                        else {
                            std::cout << "Invalid choice." << std::endl;
                            waitForKeyPress();
                            continue;
                        }
                    }
                    break;
                }

                case 4: {
                    // Check CUDA availability details
                    std::cout << "Checking CUDA availability..." << std::endl;
                    bool cuda_available = DarkChannel::isCudaAvailable();

                    if (cuda_available) {
                        std::cout << "CUDA is available and working correctly." << std::endl;

                        // Get more detailed CUDA info
                        int deviceCount = 0;
                        cudaGetDeviceCount(&deviceCount);

                        std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

                        for (int i = 0; i < deviceCount; i++) {
                            cudaDeviceProp deviceProp;
                            cudaGetDeviceProperties(&deviceProp, i);

                            std::cout << "\nDevice " << i << ": " << deviceProp.name << std::endl;
                            std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
                            std::cout << "  Total global memory: " << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
                            std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
                            std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
                            std::cout << "  Max threads dimensions: ("
                                << deviceProp.maxThreadsDim[0] << ", "
                                << deviceProp.maxThreadsDim[1] << ", "
                                << deviceProp.maxThreadsDim[2] << ")" << std::endl;
                            std::cout << "  Max grid dimensions: ("
                                << deviceProp.maxGridSize[0] << ", "
                                << deviceProp.maxGridSize[1] << ", "
                                << deviceProp.maxGridSize[2] << ")" << std::endl;
                        }
                    }
                    else {
                        std::cout << "CUDA is not available or not functioning correctly." << std::endl;
                    }

                    waitForKeyPress();
                    break;
                }

                case 5: {
                    // Performance benchmarking mode
                    std::cout << "\n=== Performance Benchmarking Mode ===" << std::endl;
                    std::cout << "This will process multiple images and export performance data for plotting." << std::endl;

                    int numImages;
                    std::cout << "Enter number of images to benchmark: ";
                    std::cin >> numImages;
                    std::cin.ignore();

                    if (numImages <= 0) {
                        std::cout << "Invalid number. Returning to main menu." << std::endl;
                        waitForKeyPress();
                        break;
                    }

                    std::vector<std::string> imagePaths;
                    std::vector<std::pair<DarkChannel::TimingInfo, DarkChannel::TimingInfo>> timingPairs;
                    std::vector<DarkChannel::TimingInfo> cudaTimings;

                    for (int i = 0; i < numImages; i++) {
                        std::cout << "\nImage " << (i + 1) << " of " << numImages << ":" << std::endl;
                        std::cout << "  1: Enter path manually" << std::endl;
                        std::cout << "  2: Use file dialog" << std::endl;

                        int imgChoice;
                        std::cout << "Choice: ";
                        std::cin >> imgChoice;
                        std::cin.ignore();

                        std::string imgPath;
                        if (imgChoice == 1) {
                            std::cout << "Enter image path: ";
                            std::getline(std::cin, imgPath);
                            if (imgPath.empty() || imgPath == "exit" || imgPath == "quit") {
                                std::cout << "Image selection canceled. Skipping this image." << std::endl;
                                i--;
                                continue;
                            }
                        }
                        else if (imgChoice == 2) {
                            imgPath = getImageFile();
                            if (imgPath.empty()) {
                                std::cout << "No file selected. Skipping this image." << std::endl;
                                i--;
                                continue;
                            }
                        }
                        else {
                            std::cout << "Invalid choice. Skipping this image." << std::endl;
                            i--;
                            continue;
                        }

                        // Verify file exists
                        if (!fileExists(imgPath)) {
                            std::cerr << "Error: File does not exist: " << imgPath << std::endl;
                            std::cout << "Skipping this image." << std::endl;
                            i--;
                            continue;
                        }

                        // Process the image silently
                        std::cout << "Processing image: " << imgPath << std::endl;

                        // Load image using custom loader or OpenCV as fallback
                        ImageU8 img;

                        // Check file extension
                        std::string ext = imgPath.substr(imgPath.find_last_of(".") + 1);
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                        try {
                            if (ext == "bmp") {
                                // Load BMP directly with our custom loader
                                img = ImageUtils::loadBMP(imgPath);
                            }
                            else {
                                // For other formats, temporarily use OpenCV just for loading
                                cv::Mat cv_img = cv::imread(imgPath);
                                if (cv_img.empty()) {
                                    std::cerr << "Error: Failed to load image: " << imgPath << std::endl;
                                    std::cout << "Skipping this image." << std::endl;
                                    i--;
                                    continue;
                                }
                                img = DarkChannel::matToImage(cv_img);
                            }

                            if (img.empty()) {
                                std::cerr << "Error: Failed to load image: " << imgPath << std::endl;
                                std::cout << "Skipping this image." << std::endl;
                                i--;
                                continue;
                            }
                        }
                        catch (const std::exception& e) {
                            std::cerr << "Exception while loading image: " << e.what() << std::endl;
                            std::cout << "Skipping this image." << std::endl;
                            i--;
                            continue;
                        }

                        // Add the image path to our list
                        imagePaths.push_back(imgPath);

                        // Resize if needed for faster processing
                        ImageU8 processImg;
                        int maxDimension = 1200;
                        if (img.width > maxDimension || img.height > maxDimension) {
                            double scale = static_cast<double>(maxDimension) /
                                (img.width > img.height ? img.width : img.height);
                            int newWidth = static_cast<int>(img.width * scale);
                            int newHeight = static_cast<int>(img.height * scale);
                            processImg = ImageUtils::customResize(img, newWidth, newHeight);
                            std::cout << "  Image resized from " << img.width << "x" << img.height
                                << " to " << processImg.width << "x" << processImg.height << std::endl;
                        }
                        else {
                            processImg = img;
                        }

                        // Process with Serial version
                        std::cout << "  Running serial version..." << std::endl;
                        try {
                            ImageU8 serial_result = DarkChannel::dehaze(processImg);
                            DarkChannel::TimingInfo serial_timing = DarkChannel::getLastTimingInfo();

                            // Process with OpenMP version
                            std::cout << "  Running OpenMP version..." << std::endl;
                            int num_threads = get_max(1, omp_get_max_threads());
                            ImageU8 openmp_result = DarkChannel::dehazeParallel(processImg, num_threads);
                            DarkChannel::TimingInfo openmp_timing = DarkChannel::getLastTimingInfo();

                            // Store the timings
                            timingPairs.push_back(std::make_pair(serial_timing, openmp_timing));

                            // Process with CUDA if available
                            bool localCudaAvailable = DarkChannel::isCudaAvailable();
                            if (localCudaAvailable) {
                                std::cout << "  Running CUDA version..." << std::endl;
                                try {
                                    ImageU8 cuda_result = DarkChannel::dehaze_cuda(processImg);
                                    DarkChannel::TimingInfo cuda_timing = DarkChannel::getLastTimingInfo();
                                    cudaTimings.push_back(cuda_timing);
                                }
                                catch (const std::exception& e) {
                                    std::cerr << "CUDA processing failed: " << e.what() << std::endl;
                                    localCudaAvailable = false;
                                }
                            }

                            std::cout << "  Image " << (i + 1) << " complete." << std::endl;
                        }
                        catch (const std::exception& e) {
                            std::cerr << "Exception during processing: " << e.what() << std::endl;
                            // Remove the path since processing failed
                            imagePaths.pop_back();
                            i--;
                            continue;
                        }
                    }

                    // Check if we successfully processed any images
                    if (imagePaths.empty() || timingPairs.empty()) {
                        std::cout << "\nNo images were successfully processed. Cannot generate benchmark results." << std::endl;
                        waitForKeyPress();
                        break;
                    }

                    // Export performance data
                    std::string csvFilename = project_path + "performance_data.csv";
                    bool hasCudaData = DarkChannel::isCudaAvailable() && !cudaTimings.empty();

                    exportPerformanceDataToCSV(
                        csvFilename,
                        imagePaths,
                        timingPairs,
                        hasCudaData,
                        hasCudaData ? &cudaTimings : nullptr
                    );

                    std::cout << "\nBenchmarking complete. Data exported to: " << csvFilename << std::endl;
                    std::cout << "You can now import this CSV file for plotting." << std::endl;

                    waitForKeyPress();
                    break;
                }

                default:
                    std::cout << "Invalid choice. Please try again." << std::endl;
                    waitForKeyPress();
                }
            }
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in main: " << e.what() << std::endl;
        waitForKeyPress();
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown exception in main" << std::endl;
        waitForKeyPress();
        return 1;
    }
}