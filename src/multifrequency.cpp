#include <SLutils/multifrequency.hpp>

#include <SLutils/fringe_analysis.hpp> // NStepPhaseShifting

#include <cmath> // std::fmod


namespace sl {

static cv::Mat equivalentPhase(cv::InputArray _phase1, cv::InputArray _phase2) {
    constexpr double twoPI = 2*CV_PI;
    
    // Get input arrays
    cv::Mat phase1 = _phase1.getMat();
    cv::Mat phase2 = _phase2.getMat();
    
    // Set output array
    cv::Mat eqPhase(phase1.size(), phase1.type());
    
    // Estimate equivalent phase as mod(phase1-phase2, 2*pi)
    const double* pphase1 = phase1.ptr<double>();
    const double* pphase2 = phase2.ptr<double>();
    double* peqPhase = eqPhase.ptr<double>();
    for (std::size_t i = 0; i < phase1.total(); i++) {
        double diff = pphase1[i] - pphase2[i];
        
        double mod = std::remainder(diff, twoPI);
        if (mod < 0) mod += twoPI;
        peqPhase[i] = mod;
    }
    
    return eqPhase;
}

static void backwardUnwrap(cv::InputArray _phase1, cv::InputOutputArray _phase2, double T1, double T2) {
    cv::Mat phase1 = _phase1.getMat();
    cv::Mat phase2 = _phase2.getMat();
    
    const double* pphase1 = phase1.ptr<double>();
    double* pphase2 = phase2.ptr<double>();
    for (std::size_t i = 0; i < phase1.total(); i++) {
        double phi2 = pphase2[i];
        
        // Estimate phase order
        double k = (T1/T2*pphase1[i] - phi2)/2/CV_PI;
        
        // Unwrap phase value
        pphase2[i] = phi2 + 2*CV_PI*cvRound(k);
    }
}

void threeFreqPhaseUnwrap(const std::vector<std::string>& impaths, cv::OutputArray _Phi,
                          const cv::Vec3i& p, const cv::Vec3i& N) {
    if (impaths.size() != (N[0]+N[1]+N[2]))
        throw std::runtime_error("threeFreqPhaseUnwrap: number of image paths and number of patterns N must match");
    
    // Get input fringe periods
    double T1 = p[0], T2 = p[1], T3 = p[2];
    // Estimate equivalent intermidate periods
    double T12 = T1*T2/std::abs(T1-T2);
    double T23 = T2*T3/std::abs(T2-T3);
    double T123 = T12*T3/std::abs(T12-T3);
    
    // Estimating wrapped phase map for each frequency
    cv::Mat phi1, phi2, phi3;
    NStepPhaseShifting({impaths.begin(), impaths.begin()+N[0]}, phi1, N[0]);
    NStepPhaseShifting({impaths.begin()+N[0], impaths.begin()+N[0]+N[1]}, phi2, N[1]);
    NStepPhaseShifting({impaths.end()-N[2], impaths.end()}, phi3, N[2]);
    
    // Estimate equivalent phase maps
    cv::Mat phi12 = equivalentPhase(phi1, phi2);
    cv::Mat phi23 = equivalentPhase(phi2, phi3);
    cv::Mat Phi123 = equivalentPhase(phi12, phi3); // Phi123 is a wide phase without discontinuities
    
    // Remove spiky noise in the equivalent phase of wider pitch
    cv::Mat Phi123m;
    Phi123.convertTo(Phi123m, CV_32F); // cv::medianBlur needs float input Mat
    cv::medianBlur(Phi123m, Phi123m, 5);
    
    double* pPhi123 = Phi123.ptr<double>();
    float* pPhi123m = Phi123m.ptr<float>();
    for (std::size_t i = 0; i < Phi123.total(); i++) {        
        // Estimate phase order difference between phase and filtered phase
        double n = (pPhi123[i] - pPhi123m[i])/2/CV_PI;
        // Estimate 2*pi multiple to remove the spike (rounding n to nearest int)
        // For pixels with no spikes rounded n must be 0 and no offset is applied
        double offset = 2*CV_PI*cvRound(n);
        
        // Correct phase value
        pPhi123[i] -= offset;
    }
    
    // Backward phase unwrapping
    backwardUnwrap(Phi123, phi23, T123, T23); // Estimate unwrapped version of phi23
    backwardUnwrap(phi23, phi12, T23, T12); // Estimate unwrapped version of phi12
    backwardUnwrap(phi12, phi3, T12, T3); // Estimate unwrapped version of phi3
    backwardUnwrap(phi3, phi2, T3, T2); // Estimate unwrapped version of phi2
    backwardUnwrap(phi2, phi1, T2, T1); // Estimate unwrapped version of phi1
    
    _Phi.assign(phi1);
}

void twoFreqPhaseUnwrap(const std::vector<std::string>& impaths, cv::OutputArray _Phi,
                        const cv::Vec3i& p, const cv::Vec3i& N) {
    if (impaths.size() != (N[0]+N[1]))
        throw std::runtime_error("twoFreqPhaseUnwrap: number of image paths and number of patterns N must match");
    
    // Get input fringe periods
    double T1 = p[0], T2 = p[1];
    // Estimate equivalent period
    double T12 = T1*T2/std::abs(T1-T2);
    
    // Estimating wrapped phase map for each frequency
    cv::Mat phi1, phi2;
    NStepPhaseShifting({impaths.begin(), impaths.begin()+N[0]}, phi1, N[0]);
    NStepPhaseShifting({impaths.begin()+N[0], impaths.end()}, phi2, N[1]);
    
    // Estimate equivalent phase map
    cv::Mat Phi12 = equivalentPhase(phi1, phi2); // Phi12 is a phase map without discontinuities
    
    // Remove spiky noise in the equivalent phase of wider pitch
    cv::Mat Phi12m;
    Phi12.convertTo(Phi12m, CV_32F); // cv::medianBlur needs float input Mat
    cv::medianBlur(Phi12m, Phi12m, 5);
    
    double* pPhi12 = Phi12.ptr<double>();
    float* pPhi12m = Phi12m.ptr<float>();
    for (std::size_t i = 0; i < Phi12.total(); i++) {        
        // Estimate phase order difference between phase and filtered phase
        double n = (pPhi12[i] - pPhi12m[i])/2/CV_PI;
        // Estimate 2*pi multiple to remove the spike (rounding n to nearest int)
        // For pixels with no spikes rounded n must be 0 and no offset is applied
        double offset = 2*CV_PI*cvRound(n);
        
        // Correct phase value
        pPhi12[i] -= offset;
    }
    
    // Backward phase unwrapping
    backwardUnwrap(Phi12, phi2, T12, T2); // Estimate unwrapped version of phi2
    backwardUnwrap(phi2, phi1, T2, T1); // Estimate unwrapped version of phi1
    
    _Phi.assign(phi1);
}

} // namespace sl
