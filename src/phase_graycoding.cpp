#include <SLutils/phase_graycoding.hpp>

#include <SLutils/fringe_analysis.hpp> // NStepPhaseShifting
#include <SLutils/graycoding.hpp> // decimalMap

#include <cmath>

void sl::phaseGraycodingUnwrap(const std::vector<std::string>& imlist_ps,
                               const std::vector<std::string>& imlist_gc,
                               cv::OutputArray _Phi, int p, int N) {
    // Estimate wrapped phase map
    cv::Mat phi;
    NStepPhaseShifting(imlist_ps, phi, N);
    
    // Estimate decimal map (phase order) with the gray patterns
    cv::Mat k;
    decimalMap(imlist_gc, k);
    k.convertTo(k, CV_64F); // convert to double*/

    // Shift and rewrap wrapped phase
    double shift = -CV_PI + CV_PI / p;
    double* phid = phi.ptr<double>();
    for (int i = 0; i < phi.total(); i++)
        phid[i] = std::atan2(std::sin(phid[i] + shift), std::cos(phid[i] + shift));

    // Estimate absolute phase map
    _Phi.create(phi.size(), CV_64F);
    cv::Mat Phi = _Phi.getMat();
    Phi = phi + 2 * CV_PI * k;

    // Shift phase back to the original values
    Phi -= shift;

    // Filter spiky noise
    cv::Mat Phim;
    Phi.convertTo(Phim, CV_32F); // cv::medianBlur needs float input Mat
    cv::medianBlur(Phim, Phim, 5);

    double* pPhi = Phi.ptr<double>();
    float* pPhim = Phim.ptr<float>();
    for (int i = 0; i < Phi.total(); i++)
        pPhi[i] -= 2 * CV_PI * cvRound((pPhi[i] - pPhim[i]) / 2 / CV_PI);
}
