#pragma once

#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>
#include <utility> // std::pair


namespace sl {

void decimalMap(const std::vector<std::string>& imlist, cv::OutputArray _dec);
void graycodeword(const std::vector<std::string>& imlist, cv::OutputArray _code_word);
cv::Mat gray2dec(cv::InputArray _code_word);
void decode(cv::InputArray _code_word, std::vector<float>& coor, cv::InputArray _mask);

} // namespace sl
