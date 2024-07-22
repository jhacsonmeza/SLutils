#pragma once

#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>

namespace sl {

void threeFreqPhaseUnwrap(const std::vector<std::string>& impaths, cv::OutputArray Phi,
                          const cv::Vec3i& p, const cv::Vec3i& N);


void twoFreqPhaseUnwrap(const std::vector<std::string>& impaths, cv::OutputArray Phi,
                        const cv::Vec3i& p, const cv::Vec3i& N);

} // namespace sl
