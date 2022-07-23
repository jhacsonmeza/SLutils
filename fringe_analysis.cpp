#include "fringe_analysis.hpp"

#include <cmath> // std::atan2, std::sqrt

void NStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase, int N)
{
    double delta = 2 * CV_PI / N;
    cv::Mat1d I = cv::imread(imgs[0], 0);
    cv::Mat1d sumIsin = I * std::sin(delta);
    cv::Mat1d sumIcos = I * std::cos(delta);

    for (int i = 1; i < imgs.size(); i++)
    {
        cv::Mat1d I = cv::imread(imgs[i], 0);
        double delta = 2 * CV_PI * (i + 1) / N;
        sumIsin += I * std::sin(delta);
        sumIcos += I * std::cos(delta);
    }

    _phase.create(sumIsin.size(), sumIsin.type());
    cv::Mat1d phase = _phase.getMat();

    double* pphase = phase.ptr<double>();
    double* psumIsin = sumIsin.ptr<double>();
    double* psumIcos = sumIcos.ptr<double>();
    for (int i = 0; i < sumIsin.total(); i++)
        pphase[i] = -std::atan2(psumIsin[i], psumIcos[i]);
}

void ThreeStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase)
{
    std::vector<cv::Mat1d> I(3);
    for (int i = 0; i < 3; i++)
        I[i] = cv::imread(imgs[i], 0);

    _phase.create(I[0].size(), I[0].type());
    cv::Mat1d phase = _phase.getMat();
    for (int i = 0; i < phase.rows; i++)
        for (int j = 0; j < phase.cols; j++)
            phase(i, j) = std::atan2(std::sqrt(3) * (I[0](i, j) - I[2](i, j)), 2 * I[1](i, j) - I[0](i, j) - I[2](i, j));
}

//void modulation(cv::InputArray, cv::OutputArray, int);