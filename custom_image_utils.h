#pragma once
#ifndef CUSTOM_IMAGE_UTILS_H
#define CUSTOM_IMAGE_UTILS_H

#include "custom_types.h"
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>

namespace ImageUtils {
    // Function to resize image using our custom Image class
    template <typename T>
    Image<T> customResize(const Image<T>& img, int newWidth, int newHeight) {
        // Simply use our existing resize method in the Image class
        return img.resize(newWidth, newHeight);
    }

    // Simple BMP file header structure
#pragma pack(push, 1)
    struct BMPHeader {
        // File header
        char signature[2] = { 'B', 'M' };
        uint32_t fileSize = 0;
        uint32_t reserved = 0;
        uint32_t dataOffset = 54;

        // Info header
        uint32_t infoSize = 40;
        int32_t width = 0;
        int32_t height = 0;
        uint16_t planes = 1;
        uint16_t bitsPerPixel = 24;
        uint32_t compression = 0;
        uint32_t imageSize = 0;
        int32_t xPixelsPerMeter = 2835; // 72 DPI
        int32_t yPixelsPerMeter = 2835; // 72 DPI
        uint32_t colorsUsed = 0;
        uint32_t importantColors = 0;
    };
#pragma pack(pop)

    // Function to save image as BMP without using OpenCV
    template <typename T>
    bool saveBMP(const Image<T>& image, const std::string& filename) {
        if (image.empty() || (image.channels != 3 && image.channels != 1)) {
            std::cerr << "Error: Image is empty or has unsupported channel count" << std::endl;
            return false;
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
            return false;
        }

        // Calculate padding - BMP rows are padded to multiples of 4 bytes
        //int padding = (4 - (image.width * 3) % 4) % 4;
        int padding = (4 - (static_cast<int>(image.width * 3) % 4)) % 4;
        int rowSize = image.width * 3 + padding;

        // Setup BMP header
        BMPHeader header;
        header.width = image.width;
        header.height = image.height;
        header.imageSize = rowSize * image.height;
        header.fileSize = header.dataOffset + header.imageSize;

        // Write header
        file.write(reinterpret_cast<char*>(&header), sizeof(BMPHeader));

        // BMP stores images bottom-up, so we start from the last row
        std::vector<unsigned char> row(rowSize, 0);

        for (int y = image.height - 1; y >= 0; y--) {
            for (int x = 0; x < image.width; x++) {
                int pixelOffset = x * 3;

                if (image.channels == 3) {
                    // BMP uses BGR order (same as our Image class)
                    row[pixelOffset] = static_cast<unsigned char>(image.at(y, x, 0));     // B
                    row[pixelOffset + 1] = static_cast<unsigned char>(image.at(y, x, 1)); // G
                    row[pixelOffset + 2] = static_cast<unsigned char>(image.at(y, x, 2)); // R
                }
                else {
                    // Grayscale image - replicate to all channels
                    unsigned char value = static_cast<unsigned char>(image.at(y, x, 0));
                    row[pixelOffset] = value;
                    row[pixelOffset + 1] = value;
                    row[pixelOffset + 2] = value;
                }
            }

            // Write row with padding
            file.write(reinterpret_cast<char*>(row.data()), rowSize);
        }

        file.close();
        return true;
    }

    // Function to load an image from BMP without using OpenCV
    ImageU8 loadBMP(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
            return ImageU8();
        }

        // Read BMP header
        BMPHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(BMPHeader));

        // Verify BMP signature
        if (header.signature[0] != 'B' || header.signature[1] != 'M') {
            std::cerr << "Error: Not a valid BMP file: " << filename << std::endl;
            return ImageU8();
        }

        // Check if supported format (24-bit)
        if (header.bitsPerPixel != 24) {
            std::cerr << "Error: Only 24-bit BMP files are supported. File has "
                << header.bitsPerPixel << " bits per pixel." << std::endl;
            return ImageU8();
        }

        // Calculate padding - BMP rows are padded to multiples of 4 bytes
        int padding = (4 - (header.width * 3) % 4) % 4;
        int rowSize = header.width * 3 + padding;

        // Create image
        ImageU8 image(header.width, header.height, 3);

        // Seek to image data
        file.seekg(header.dataOffset, std::ios::beg);

        // BMP stores images bottom-up, so we start from the last row
        std::vector<unsigned char> row(rowSize);

        for (int y = header.height - 1; y >= 0; y--) {
            file.read(reinterpret_cast<char*>(row.data()), rowSize);

            for (int x = 0; x < header.width; x++) {
                int pixelOffset = x * 3;

                // BMP uses BGR order (same as our Image class)
                image.at(y, x, 0) = row[pixelOffset];     // B
                image.at(y, x, 1) = row[pixelOffset + 1]; // G
                image.at(y, x, 2) = row[pixelOffset + 2]; // R
            }
        }

        file.close();
        return image;
    }

    // Function to generate an HTML viewer for side-by-side image comparison
    void generateHtmlViewer(const std::vector<std::string>& imagePaths,
        const std::vector<std::string>& titles,
        const std::string& outputHtmlPath) {
        std::ofstream htmlFile(outputHtmlPath);
        if (!htmlFile.is_open()) {
            std::cerr << "Error: Could not create HTML file: " << outputHtmlPath << std::endl;
            return;
        }

        // Start HTML document
        htmlFile << "<!DOCTYPE html>\n"
            << "<html lang=\"en\">\n"
            << "<head>\n"
            << "    <meta charset=\"UTF-8\">\n"
            << "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            << "    <title>Image Dehaze Results</title>\n"
            << "    <style>\n"
            << "        body {\n"
            << "            font-family: Arial, sans-serif;\n"
            << "            margin: 20px;\n"
            << "            background-color: #f5f5f5;\n"
            << "        }\n"
            << "        h1 {\n"
            << "            color: #333;\n"
            << "            text-align: center;\n"
            << "        }\n"
            << "        .image-container {\n"
            << "            display: flex;\n"
            << "            flex-wrap: wrap;\n"
            << "            justify-content: center;\n"
            << "            gap: 20px;\n"
            << "            margin-bottom: 30px;\n"
            << "        }\n"
            << "        .image-box {\n"
            << "            background-color: white;\n"
            << "            padding: 10px;\n"
            << "            border-radius: 5px;\n"
            << "            box-shadow: 0 2px 5px rgba(0,0,0,0.1);\n"
            << "        }\n"
            << "        .image-box img {\n"
            << "            max-width: 100%;\n"
            << "            height: auto;\n"
            << "            display: block;\n"
            << "        }\n"
            << "        .image-title {\n"
            << "            margin-top: 10px;\n"
            << "            text-align: center;\n"
            << "            font-weight: bold;\n"
            << "        }\n"
            << "    </style>\n"
            << "</head>\n"
            << "<body>\n"
            << "    <h1>Image Dehazing Results</h1>\n";

        // Add image containers
        htmlFile << "    <div class=\"image-container\">\n";

        for (size_t i = 0; i < imagePaths.size(); i++) {
            htmlFile << "        <div class=\"image-box\">\n"
                << "            <img src=\"file:///" << imagePaths[i] << "\" alt=\"" << titles[i] << "\">\n"
                << "            <div class=\"image-title\">" << titles[i] << "</div>\n"
                << "        </div>\n";
        }

        htmlFile << "    </div>\n";

        // Close HTML document
        htmlFile << "</body>\n"
            << "</html>\n";

        htmlFile.close();

        std::cout << "HTML viewer created: " << outputHtmlPath << std::endl;
        std::cout << "Open this file in any web browser to view the results side by side." << std::endl;
    }

    // Function to display image results without using OpenCV
    void displayResults(const std::string& outputDir, const std::string& imageName,
        const std::string& originalPath, const std::string& serialPath,
        const std::string& openmpPath, const std::string& cudaPath = "") {

        std::vector<std::string> imagePaths;
        std::vector<std::string> titles;

        // Add all available image paths and titles
        imagePaths.push_back(originalPath);
        titles.push_back("Original Image");

        imagePaths.push_back(serialPath);
        titles.push_back("Serial Dehazed");

        imagePaths.push_back(openmpPath);
        titles.push_back("OpenMP Dehazed");

        if (!cudaPath.empty()) {
            imagePaths.push_back(cudaPath);
            titles.push_back("CUDA Dehazed");
        }

        // Generate HTML file path (use absolute paths)
        std::string htmlFilePath = outputDir + "result_" + imageName + ".html";

        // Make paths absolute for proper display in HTML
        for (auto& path : imagePaths) {
            // Convert to absolute path if necessary
            if (path.find(':') == std::string::npos) {
                // This is a relative path, make it absolute
                char absPath[1024];
#ifdef _WIN32
                _fullpath(absPath, path.c_str(), 1024);
#else
                realpath(path.c_str(), absPath);
#endif
                path = std::string(absPath);
            }

            // Replace backslashes with forward slashes for HTML
            std::replace(path.begin(), path.end(), '\\', '/');
        }

        // Generate and open HTML viewer
        generateHtmlViewer(imagePaths, titles, htmlFilePath);

        // Try to automatically open the HTML file in the default browser
#ifdef _WIN32
        std::string command = "start \"\" \"" + htmlFilePath + "\"";
        system(command.c_str());
#elif __APPLE__
        std::string command = "open \"" + htmlFilePath + "\"";
        system(command.c_str());
#else // Linux
        std::string command = "xdg-open \"" + htmlFilePath + "\"";
        system(command.c_str());
#endif

        std::cout << "Displaying results in your default web browser..." << std::endl;
    }

    // Function to get image dimensions without using OpenCV
    std::pair<int, int> getImageDimensions(const std::string& imagePath) {
        // Check file extension to determine format
        std::string ext = imagePath.substr(imagePath.find_last_of(".") + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == "bmp") {
            // BMP handling (keep your existing code)
            std::ifstream file(imagePath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open BMP file: " << imagePath << std::endl;
                return { 0, 0 };
            }

            BMPHeader header;
            file.read(reinterpret_cast<char*>(&header), sizeof(BMPHeader));
            file.close();

            if (header.signature[0] != 'B' || header.signature[1] != 'M') {
                std::cerr << "Error: Not a valid BMP file: " << imagePath << std::endl;
                return { 0, 0 };
            }

            return { header.width, header.height };
        }
        else {
            // For other formats, use OpenCV to get dimensions
            // This is consistent with your approach of using OpenCV for reading non-BMP formats
            cv::Mat img = cv::imread(imagePath);
            if (img.empty()) {
                std::cerr << "Error: Could not determine dimensions of: " << imagePath << std::endl;
                return { 0, 0 };
            }

            return { img.cols, img.rows };
        }
    }
}

#endif // CUSTOM_IMAGE_UTILS_H