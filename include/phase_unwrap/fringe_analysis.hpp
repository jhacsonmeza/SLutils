#pragma once

#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>


namespace sl {

void NStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase, int N);

void NStepPhaseShifting_modulation(const std::vector<std::string>& imgs, cv::OutputArray _phase,
                                   cv::OutputArray _data_modulation, int N);

void ThreeStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase);

void ThreeStepPhaseShifting_modulation(const std::vector<std::string>& imgs, cv::OutputArray _phase,
                                       cv::OutputArray _data_modulation, int N);

} // namespace sl
