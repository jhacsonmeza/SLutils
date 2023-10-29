#pragma once

#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>

void NStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase, int N);
void ThreeStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase);
void modulation(const std::vector<std::string>& imgs, cv::OutputArray _data_modulation, int N);
