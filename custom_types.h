#pragma once
#ifndef CUSTOM_TYPES_H
#define CUSTOM_TYPES_H

#include <vector>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>

// Custom Image class to replace OpenCV Mat
template <typename T>
class Image {
public:
    // Constructor for empty image
    Image() : data(nullptr), width(0), height(0), channels(0), step(0), ownsData(true) {}

    // Constructor with dimensions
    Image(int width, int height, int channels = 1)
        : width(width), height(height), channels(channels), ownsData(true) {
        step = width * channels;
        data = new T[width * height * channels]();  // Initialized with zeros
    }

    // Copy constructor
    Image(const Image& other)
        : width(other.width), height(other.height), channels(other.channels), step(other.step), ownsData(true) {
        const size_t totalSize = width * height * channels;
        data = new T[totalSize];
        std::memcpy(data, other.data, totalSize * sizeof(T));
    }

    // Move constructor
    Image(Image&& other) noexcept
        : data(other.data), width(other.width), height(other.height),
        channels(other.channels), step(other.step), ownsData(other.ownsData) {
        other.data = nullptr;
        other.ownsData = false;
    }

    // Assignment operator
    Image& operator=(const Image& other) {
        if (this != &other) {
            if (ownsData && data) {
                delete[] data;
            }

            width = other.width;
            height = other.height;
            channels = other.channels;
            step = other.step;
            ownsData = true;

            const size_t totalSize = width * height * channels;
            data = new T[totalSize];
            std::memcpy(data, other.data, totalSize * sizeof(T));
        }
        return *this;
    }

    // Move assignment operator
    Image& operator=(Image&& other) noexcept {
        if (this != &other) {
            if (ownsData && data) {
                delete[] data;
            }

            data = other.data;
            width = other.width;
            height = other.height;
            channels = other.channels;
            step = other.step;
            ownsData = other.ownsData;

            other.data = nullptr;
            other.ownsData = false;
        }
        return *this;
    }

    // Destructor
    ~Image() {
        if (ownsData && data) {
            delete[] data;
        }
    }

    // Create a new image with the same dimensions
    Image clone() const {
        Image result(width, height, channels);
        const size_t totalSize = width * height * channels;
        std::memcpy(result.data, data, totalSize * sizeof(T));
        return result;
    }

    // Check if image is empty
    bool empty() const {
        return data == nullptr || width == 0 || height == 0;
    }

    // Get data pointer for a specific pixel
    T* ptr(int y, int x = 0) {
        return data + (y * step + x * channels);
    }

    // Get const data pointer for a specific pixel
    const T* ptr(int y, int x = 0) const {
        return data + (y * step + x * channels);
    }

    // Element access with bounds checking
    T& at(int y, int x, int c = 0) {
        if (y < 0 || y >= height || x < 0 || x >= width || c < 0 || c >= channels) {
            throw std::out_of_range("Image index out of range");
        }
        return data[y * step + x * channels + c];
    }

    // Const element access with bounds checking
    const T& at(int y, int x, int c = 0) const {
        if (y < 0 || y >= height || x < 0 || x >= width || c < 0 || c >= channels) {
            throw std::out_of_range("Image index out of range");
        }
        return data[y * step + x * channels + c];
    }

    // Direct memory access for optimization
    T* get_data() { return data; }
    const T* get_data() const { return data; }

    // Create image from existing data (no copy)
    static Image createFromData(T* data, int width, int height, int channels) {
        Image img;
        img.data = data;
        img.width = width;
        img.height = height;
        img.channels = channels;
        img.step = width * channels;
        img.ownsData = false;  // The caller owns the data
        return img;
    }

    // Type conversion (similar to OpenCV's convertTo)
    template <typename U>
    Image<U> convertTo(double scale = 1.0, double offset = 0.0) const {
        Image<U> result(width, height, channels);
        const size_t totalSize = width * height * channels;

        for (size_t i = 0; i < totalSize; ++i) {
            result.data[i] = static_cast<U>(data[i] * scale + offset);
        }

        return result;
    }

    // Divide all elements by a scalar
    Image& operator/=(T scalar) {
        const size_t totalSize = width * height * channels;
        for (size_t i = 0; i < totalSize; ++i) {
            data[i] /= scalar;
        }
        return *this;
    }

    // Element-wise matrix multiplication
    Image mul(const Image& other) const {
        if (width != other.width || height != other.height || channels != other.channels) {
            throw std::invalid_argument("Images must have the same dimensions for multiplication");
        }

        Image result(width, height, channels);
        const size_t totalSize = width * height * channels;

        for (size_t i = 0; i < totalSize; ++i) {
            result.data[i] = data[i] * other.data[i];
        }

        return result;
    }

    // Resize image with bilinear interpolation
    Image resize(int newWidth, int newHeight) const {
        Image result(newWidth, newHeight, channels);

        double scaleX = static_cast<double>(width) / newWidth;
        double scaleY = static_cast<double>(height) / newHeight;

        for (int y = 0; y < newHeight; ++y) {
            for (int x = 0; x < newWidth; ++x) {
                double srcX = x * scaleX;
                double srcY = y * scaleY;

                int x1 = static_cast<int>(srcX);
                int y1 = static_cast<int>(srcY);
                int x2 = std::min(x1 + 1, width - 1);
                int y2 = std::min(y1 + 1, height - 1);

                double wx = srcX - x1;
                double wy = srcY - y1;

                for (int c = 0; c < channels; ++c) {
                    double topLeft = at(y1, x1, c);
                    double topRight = at(y1, x2, c);
                    double bottomLeft = at(y2, x1, c);
                    double bottomRight = at(y2, x2, c);

                    double top = topLeft * (1 - wx) + topRight * wx;
                    double bottom = bottomLeft * (1 - wx) + bottomRight * wx;

                    result.at(y, x, c) = static_cast<T>(top * (1 - wy) + bottom * wy);
                }
            }
        }

        return result;
    }

    int width;
    int height;
    int channels;
    int step;      // Step between rows (in elements)
    T* data;
    bool ownsData; // Whether this object owns the data pointer
};

// Define common image types
using ImageU8 = Image<unsigned char>;
using ImageF32 = Image<float>;
using ImageF64 = Image<double>;

// Type recognition traits
template <typename T>
struct ImageTypeTraits {};

template <>
struct ImageTypeTraits<unsigned char> { static constexpr int depth = 0; };

template <>
struct ImageTypeTraits<float> { static constexpr int depth = 1; };

template <>
struct ImageTypeTraits<double> { static constexpr int depth = 2; };

// Simple 3-element color/vector class (similar to cv::Vec3)
template <typename T>
struct Vec3 {
    T val[3];

    Vec3() : val{ 0, 0, 0 } {}
    Vec3(T v0, T v1, T v2) : val{ v0, v1, v2 } {}

    T& operator[](int i) { return val[i]; }
    const T& operator[](int i) const { return val[i]; }
};

using Vec3b = Vec3<unsigned char>;
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;

// Split and merge functions
template <typename T>
void split(const Image<T>& src, Image<T>* channels, int numChannels) {
    if (src.channels < numChannels) {
        throw std::invalid_argument("Source image has fewer channels than requested");
    }

    for (int c = 0; c < numChannels; ++c) {
        channels[c] = Image<T>(src.width, src.height, 1);

        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                channels[c].at(y, x) = src.at(y, x, c);
            }
        }
    }
}

template <typename T>
void merge(const Image<T>* channels, int numChannels, Image<T>& dst) {
    if (channels[0].empty()) {
        throw std::invalid_argument("Input channels cannot be empty");
    }

    int width = channels[0].width;
    int height = channels[0].height;

    dst = Image<T>(width, height, numChannels);

    for (int c = 0; c < numChannels; ++c) {
        if (channels[c].width != width || channels[c].height != height || channels[c].channels != 1) {
            throw std::invalid_argument("All channels must have the same dimensions and be single-channel");
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                dst.at(y, x, c) = channels[c].at(y, x);
            }
        }
    }
}

// Box filter function (replacement for cv::blur)
template <typename T>
void boxFilter(const Image<T>& src, Image<T>& dst, int ksize) {
    if (ksize <= 0 || (ksize % 2) == 0) {
        throw std::invalid_argument("Kernel size must be positive and odd");
    }

    int radius = ksize / 2;
    dst = Image<T>(src.width, src.height, src.channels);

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            for (int c = 0; c < src.channels; ++c) {
                T sum = 0;
                int count = 0;

                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int ny = y + dy;
                        int nx = x + dx;

                        if (ny >= 0 && ny < src.height && nx >= 0 && nx < src.width) {
                            sum += src.at(ny, nx, c);
                            count++;
                        }
                    }
                }

                dst.at(y, x, c) = sum / count;
            }
        }
    }
}

// Find minimum and maximum values in an image
template <typename T>
void minMaxLoc(const Image<T>& src, T* minVal, T* maxVal) {
    if (src.empty()) {
        throw std::invalid_argument("Image is empty");
    }

    T minValue = src.at(0, 0);
    T maxValue = src.at(0, 0);

    const size_t totalSize = src.width * src.height * src.channels;
    for (size_t i = 0; i < totalSize; ++i) {
        minValue = std::min(minValue, src.data[i]);
        maxValue = std::max(maxValue, src.data[i]);
    }

    if (minVal) *minVal = minValue;
    if (maxVal) *maxVal = maxValue;
}

// Element-wise maximum of two images or image and scalar
template <typename T>
void max(const Image<T>& src1, const Image<T>& src2, Image<T>& dst) {
    if (src1.width != src2.width || src1.height != src2.height || src1.channels != src2.channels) {
        throw std::invalid_argument("Images must have the same dimensions");
    }

    dst = Image<T>(src1.width, src1.height, src1.channels);

    const size_t totalSize = src1.width * src1.height * src1.channels;
    for (size_t i = 0; i < totalSize; ++i) {
        dst.data[i] = std::max(src1.data[i], src2.data[i]);
    }
}

template <typename T>
void max(const Image<T>& src1, T scalar, Image<T>& dst) {
    dst = Image<T>(src1.width, src1.height, src1.channels);

    const size_t totalSize = src1.width * src1.height * src1.channels;
    for (size_t i = 0; i < totalSize; ++i) {
        dst.data[i] = std::max(src1.data[i], scalar);
    }
}

// Element-wise division of two images
template <typename T>
void divide(const Image<T>& src1, const Image<T>& src2, Image<T>& dst) {
    if (src1.width != src2.width || src1.height != src2.height || src1.channels != src2.channels) {
        throw std::invalid_argument("Images must have the same dimensions");
    }

    dst = Image<T>(src1.width, src1.height, src1.channels);

    const size_t totalSize = src1.width * src1.height * src1.channels;
    for (size_t i = 0; i < totalSize; ++i) {
        // Avoid division by zero
        dst.data[i] = src2.data[i] != 0 ? src1.data[i] / src2.data[i] : src1.data[i];
    }
}

// Create an image filled with zeros
template <typename T>
Image<T> zeros(int width, int height, int channels = 1) {
    return Image<T>(width, height, channels);
}

#endif // CUSTOM_TYPES_H