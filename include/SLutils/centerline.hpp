#pragma once

#include <opencv2/core/mat.hpp>
#include <string>


namespace sl {

cv::Point seedPoint(const std::string& fn_clx, const std::string& fn_cly, cv::InputArray _mask);

void spatialUnwrap(cv::InputArray _phased, const cv::Point p0, cv::InputArray _mask, cv::OutputArray _Phi);

} // namespace sl
