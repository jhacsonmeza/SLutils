#include <SLutils/fringe_analysis.hpp>

#include <iostream>
#include <algorithm> // std::sort
#include <filesystem> // std::filesystem
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // cv::imshow, cv::waitKey, cv::destroyAllWindows

#ifdef HAVE_CUDA
#include <opencv2/core/cuda.hpp>
#endif


int main(int argc, char* argv[]) {
    // Path to the images
    std::filesystem::path imgs_path{argv[1]};
    
    // Creating a vector with all the image paths
    std::vector<std::string> im_files;
    for (const auto& p : std::filesystem::directory_iterator(imgs_path))
        im_files.push_back(p.path().string());
    std::sort(im_files.begin(), im_files.end());
    
    
    // ------------------------------- Wrapped phase and data modulation estimation
#ifdef HAVE_CUDA
    std::cout<<"This example is running in CUDA\n\n";
    cv::cuda::GpuMat phi_d, modu_d;
    sl::NStepPhaseShifting_modulation(im_files, phi_d, modu_d, 8);
    
    // Move to CPU
    cv::Mat phi, modu;
    phi_d.download(phi);
    modu_d.download(modu);
#else
    std::cout<<"This example is running in CPU\n\n";
    cv::Mat phi, modu;
    sl::NStepPhaseShifting_modulation(im_files, phi, modu, 8);
#endif


    // ------------------------------- Show wrapped phase map results
    std::cout<<"phi[500:503, 500:503]:\n"<<phi(cv::Range(500,503), cv::Range(500,503))<<"\n";

    try {
        // Normalize array to the range [0,1] for visualization
        cv::normalize(phi, phi, 0, 1, cv::NORM_MINMAX);
        
        // Create and set window size
        cv::namedWindow("Wrapped phase map", cv::WINDOW_NORMAL);
        cv::resizeWindow("Wrapped phase map", 896, 717);
        // Show the image
        cv::imshow("Wrapped phase map", phi);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const cv::Exception& e) {
        std::cout<<"Unable to use cv::imshow, got the error: "<<e.what();
        std::cout<<"Saving the output wrapped phase map as 'ps_wrapped_phase.png'\n\n";

        // Convert float array to uint8
        phi.convertTo(phi, CV_8U, 255);
        // Write image
        cv::imwrite("ps_wrapped_phase.png", phi);
    }
    
    
    // ------------------------------- Show data modulation results
    std::cout<<"modu[500:503, 500:503]:\n"<<modu(cv::Range(500,503), cv::Range(500,503))<<"\n";

    try {
        // modulation map is already in the [0,1] range and don't need
        // normalization for visualization
        
        // Create and set window size
        cv::namedWindow("Data modulation", cv::WINDOW_NORMAL);
        cv::resizeWindow("Data modulation", 896, 717);
        // Show the image
        cv::imshow("Data modulation", modu);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const cv::Exception& e) {
        std::cout<<"Unable to use cv::imshow, got the error: "<<e.what();
        std::cout<<"Saving the output modulation map as 'ps_modulation.png'\n";
        
        // Convert float array to uint8
        modu.convertTo(modu, CV_8U, 255);
        // Write image
        cv::imwrite("ps_modulation.png", modu);
    }
}
