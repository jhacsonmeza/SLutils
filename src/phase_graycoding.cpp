#include <SLutils/phase_graycoding.hpp>

#include <SLutils/fringe_analysis.hpp> // NStepPhaseShifting
#include <SLutils/graycoding.hpp> // decimalMap

#include <cmath>

void sl::phaseGraycodingUnwrap(const std::vector<std::string>& impaths_ps,
                               const std::vector<std::string>& impaths_gc,
                               cv::OutputArray _Phi, int p, int N) {
    // Estimate wrapped phase map
    cv::Mat phi;
    NStepPhaseShifting(impaths_ps, phi, N);
    
    // Estimate decimal map (phase order) with the gray patterns
    cv::Mat k;
    decimalMap(impaths_gc, k);
    k.convertTo(k, CV_64F); // convert to double

    // Shift and rewrap wrapped phase
    double shift = -CV_PI + CV_PI/p;
    double* phid = phi.ptr<double>();
    for (std::size_t i = 0; i < phi.total(); i++) {
        double phi_shift = phid[i] + shift; // shifted phase value
        phid[i] = std::atan2(std::sin(phi_shift), std::cos(phi_shift));
    }

    // Estimate absolute phase map
    _Phi.create(phi.size(), phi.type());
    cv::Mat Phi = _Phi.getMat();
    Phi = phi + 2*CV_PI*k;

    // Shift phase back to the original values
    Phi -= shift;

    // Filter spiky noise
    cv::Mat Phim;
    Phi.convertTo(Phim, CV_32F); // cv::medianBlur needs float input Mat
    cv::medianBlur(Phim, Phim, 5);

    double* pPhi = Phi.ptr<double>();
    float* pPhim = Phim.ptr<float>();
    for (std::size_t i = 0; i < Phi.total(); i++) {        
        // Estimate phase order difference between phase and filtered phase
        double n = (pPhi[i] - pPhim[i])/2/CV_PI;
        // Estimate 2*pi multiple to remove the spike (rounding n to nearest int)
        // For pixels with no spikes rounded n must be 0 and no offset is applied
        double offset = 2*CV_PI*cvRound(n);
        
        // Correct phase value
        pPhi[i] -= offset;
    }
}
