#pragma once

#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>

void graycodeword(const std::vector<std::string>& imlist, cv::OutputArray _code_word);
cv::Mat gray2dec(cv::InputArray _code_word);
void decode(cv::InputArray _code_word, std::vector<float>& coor, cv::InputArray _mask);
