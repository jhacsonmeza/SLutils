#ifndef FRINGE_ANALYSIS_HPP
#define FRINGE_ANALYSIS_HPP

#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>

void NStepPhaseShifting(const std::vector<std::string>&, cv::OutputArray, int);
void ThreeStepPhaseShifting(const std::vector<std::string>&, cv::OutputArray);
//void modulation(cv::InputArray, cv::OutputArray, int);

#endif