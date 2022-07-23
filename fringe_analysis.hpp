#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>

void NStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase, int N);
void ThreeStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase);
//void modulation(cv::InputArray, cv::OutputArray, int);