#pragma once

#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>


namespace sl {

void phaseGraycodingUnwrap(const std::vector<std::string>& imlist_ps,
                           const std::vector<std::string>& imlist_gc,
                           cv::OutputArray _Phi, int p, int N);

} // namespace sl
