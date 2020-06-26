#ifndef PHASE_GRAYCODING_HPP
#define PHASE_GRAYCODING_HPP

#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>

void phaseGraycodingUnwrap(const std::vector<std::string>&, const std::vector<std::string>&, cv::OutputArray, int, int);

#endif