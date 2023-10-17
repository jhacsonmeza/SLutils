#include <phase_unwrap/phase_graycoding.hpp>

#include <phase_unwrap/fringe_analysis.hpp> // NStepPhaseShifting
#include <phase_unwrap/graycoding.hpp> // codeword, grayToDec

#include <cmath>

void phaseGraycodingUnwrap(const std::vector<std::string>& imlist_ps, const std::vector<std::string>& imlist_gc,
cv::OutputArray _Phi, int p, int N)
{
    // Estimate wrapped phase map
    cv::Mat phi;
    NStepPhaseShifting(imlist_ps, phi, N);

    // Estimate code words
    cv::Mat code_word;
    codeword(imlist_gc, code_word);

    // Estimate fringe order with the codeword
    cv::Mat k = grayToDec(code_word); // return int32 Mat
    k.convertTo(k, CV_64F); // convert to double

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
