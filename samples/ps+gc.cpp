#include <phase_unwrap/phase_graycoding.hpp>

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
     
     
    int N = 18, p = 18;
    std::vector<std::string> im_files_ps(im_files.begin(), im_files.begin() + N);
	std::vector<std::string> im_files_gc(im_files.begin() + N, im_files.end());

#ifdef HAVE_CUDA
    std::cout<<"This example is running in CUDA\n\n";
    cv::cuda::GpuMat Phi_d;
	sl::phaseGraycodingUnwrap(im_files_ps, im_files_gc, Phi_d, p, N);
    
    // Move to CPU
    cv::Mat Phi;
    Phi_d.download(Phi);
#else
    std::cout<<"This example is running in CPU\n\n";
	cv::Mat Phi;
	sl::phaseGraycodingUnwrap(im_files_ps, im_files_gc, Phi, p, N);
#endif

    
    // ------------------------------- Unwrapped phase results
    std::cout<<"Phi[500:503, 500:503]:\n"<<Phi(cv::Range(500,503), cv::Range(500,503))<<"\n";

    try {
        // Normalize array to the range [0,1] for visualization
        cv::normalize(Phi, Phi, 0, 1, cv::NORM_MINMAX);
        
	    // Create and set window size
        cv::namedWindow("Absolute unwrapped phase map", cv::WINDOW_NORMAL);
        cv::resizeWindow("Absolute unwrapped phase map", 896, 717);
        // Show the image
        cv::imshow("Absolute unwrapped phase map", Phi);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const cv::Exception& e) {
        std::cout<<"Unable to use cv::imshow, got the error: "<<e.what();
        std::cout<<"Saving the output unwrapped phase map as 'ps+gc_unwrapped_phase.png'\n\n";
        
        // Convert float array to uint8
        Phi.convertTo(Phi, CV_8U, 255);
        // Write image
        cv::imwrite("ps+gc_unwrapped_phase.png", Phi);
    }
}
