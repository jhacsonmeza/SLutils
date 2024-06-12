#pragma once

#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>


namespace sl {

void phaseGraycodingUnwrap(const std::vector<std::string>& impaths_ps,
                           const std::vector<std::string>& impaths_gc,
                           cv::OutputArray Phi, int p, int N);

} // namespace sl
