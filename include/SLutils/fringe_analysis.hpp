#pragma once

#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>


namespace sl {

void NStepPhaseShifting(const std::vector<std::string>& impaths, cv::OutputArray phase, int N);

void NStepPhaseShifting_modulation(const std::vector<std::string>& impaths, cv::OutputArray phase,
                                   cv::OutputArray data_modulation, int N);

void ThreeStepPhaseShifting(const std::vector<std::string>& impaths, cv::OutputArray phase);

void ThreeStepPhaseShifting_modulation(const std::vector<std::string>& impaths, cv::OutputArray phase,
                                       cv::OutputArray data_modulation);

} // namespace sl
