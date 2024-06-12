#pragma once

#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>


namespace sl {

void decimalMap(const std::vector<std::string>& impaths, cv::OutputArray dec);

void graycodeword(const std::vector<std::string>& impaths, cv::OutputArray code_word);

void gray2dec(cv::InputArray code_word, cv::OutputArray dec);

} // namespace sl
