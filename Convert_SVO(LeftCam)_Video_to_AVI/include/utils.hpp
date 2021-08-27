#pragma once
#include <sys/types.h>
#include <sys/stat.h>

static bool exit_app = false;

// Handle the CTRL-C keyboard signal
#include <signal.h>

void nix_exit_handler(int s) {
    exit_app = true;
}

// Set the function to handle the CTRL-C
void SetCtrlHandler() {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = nix_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
}

// Display progress bar
void ProgressBar(float ratio, unsigned int w) {
    unsigned int c = ratio * w;
    for (unsigned int x = 0; x < c; x++) std::cout << "=";
    for (unsigned int x = c; x < w; x++) std::cout << " ";
    std::cout << (int) (ratio * 100) << "% ";
    std::cout << "\r" << std::flush;
}

// If the current project uses openCV
#if defined (__OPENCV_ALL_HPP__) || defined(OPENCV_ALL_HPP)
// Conversion function between sl::Mat and cv::Mat
cv::Mat slMat2cvMat(sl::Mat &input) {
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}
#endif