#pragma once

#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>

void phaseGraycodingUnwrap(const std::vector<std::string>& imlist_ps, const std::vector<std::string>& imlist_gc, cv::OutputArray _Phi, int p, int N);