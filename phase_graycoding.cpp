#include "phase_graycoding.hpp"

#include "fringe_analysis.hpp" // NStepPhaseShifting
#include "graycoding.hpp" // codeword, grayToDec

#include <cmath>

void phaseGraycodingUnwrap(const std::vector<std::string>& imlist_ps, const std::vector<std::string>& imlist_gc,
cv::OutputArray _Phi, int p, int N)
{
    // Estimate wrapped phase map
    cv::Mat1d phi;
    NStepPhaseShifting(imlist_ps, phi, N);

    // Estimate code words
    cv::Mat code_word;
    codeword(imlist_gc, code_word);

    // Estimate fringe order with codeword
    cv::Mat1d k = grayToDec(code_word);

    // Shift and rewrap wrapped phase
    double shift = -CV_PI + CV_PI / p;
    double* phid = (double*)phi.data;
    for (int i = 0; i < phi.total(); i++)
        phid[i] = std::atan2(std::sin(phid[i] + shift), std::cos(phid[i] + shift));

    // Estimate absolute phase map
    _Phi.create(phi.size(), CV_64F);
    cv::Mat1d Phi = _Phi.getMat();
    Phi = phi + 2 * CV_PI * k;

    // Shift phase back to the original values
    Phi -= shift;

    // Filter spiky noise
    cv::Mat1f Phim;
    cv::medianBlur((cv::Mat_<float>)Phi, Phim, 5);

    double* pPhi = (double*)Phi.data;
    float* pPhim = (float*)Phim.data;
    for (int i = 0; i < Phi.total(); i++)
        pPhi[i] -= 2 * CV_PI * cvRound((pPhi[i] - pPhim[i]) / 2 / CV_PI);
}