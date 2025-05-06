//#pragma once
//#ifndef IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H
//#define IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H

//#include "custom_types.h"
//#include <stdexcept>
//#include <opencv2/opencv.hpp> // Keep this only for IO operations, not algorithms

/*namespace DarkChannel {
    // Error checking macro
#define __Assert__(x,msg)\
    do\
    {\
        if(!(x)){throw std::runtime_error((msg));}\
    }while(false)

    // Structure to store timing information
    struct TimingInfo {
        double darkChannelTime = 0.0;
        double atmosphericLightTime = 0.0;
        double transmissionTime = 0.0;
        double refinementTime = 0.0;
        double reconstructionTime = 0.0;
        double totalTime = 0.0;
    };

    // Convert OpenCV Mat to custom image format (only for IO purposes)
    ImageU8 matToImage(const cv::Mat& mat);

    // Convert custom image format back to OpenCV Mat (only for IO purposes)
    cv::Mat imageToMat(const ImageU8& img);

    // Serial implementation
    ImageU8 dehaze(const ImageU8& img);

    // OpenMP implementation
    ImageU8 dehazeParallel(const ImageU8& img, int numThreads = 1);

    // CUDA implementation
    ImageU8 dehaze_cuda(const ImageU8& img);

    // Function to get the last timing information
    TimingInfo getLastTimingInfo();

    // Function to check CUDA availability with proper error handling
    bool isCudaAvailable();
}

#endif
*/

#pragma once
#ifndef IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H
#define IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H

#include "custom_types.h"
#include <stdexcept>
#include <opencv2/opencv.hpp> // Keep this only for IO operations, not algorithms

namespace DarkChannel {
    // Error checking macro
#define __Assert__(x,msg)\
    do\
    {\
        if(!(x)){throw std::runtime_error((msg));}\
    }while(false)

    // Structure to store timing information
    struct TimingInfo {
        double darkChannelTime = 0.0;
        double atmosphericLightTime = 0.0;
        double transmissionTime = 0.0;
        double refinementTime = 0.0;
        double reconstructionTime = 0.0;
        double totalTime = 0.0;
    };

    // Global timing info that can be accessed from outside
    extern TimingInfo lastTimingInfo;

    // Convert OpenCV Mat to custom image format (only for IO purposes)
    ImageU8 matToImage(const cv::Mat& mat);

    // Convert custom image format back to OpenCV Mat (only for IO purposes)
    cv::Mat imageToMat(const ImageU8& img);

    // Serial implementation
    ImageU8 dehaze(const ImageU8& img);

    // OpenMP implementation
    ImageU8 dehazeParallel(const ImageU8& img, int numThreads = 1);

    // CUDA implementation
    ImageU8 dehaze_cuda(const ImageU8& img);

    // Function to get the last timing information
    TimingInfo getLastTimingInfo();

    // Function to check CUDA availability with proper error handling
    bool isCudaAvailable();
}

#endif //IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H