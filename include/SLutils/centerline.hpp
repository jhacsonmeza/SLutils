#pragma once

#include <opencv2/core/mat.hpp>
#include <string>


namespace sl {

cv::Point seedPoint(const std::string& fn_clx, const std::string& fn_cly, cv::InputArray mask);

void spatialUnwrap(cv::InputArray phased, const cv::Point p0, cv::InputArray mask, cv::OutputArray Phi);

} // namespace sl
