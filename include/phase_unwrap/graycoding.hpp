#pragma once

#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>

cv::Mat grayToDec(cv::InputArray _code_word);
void codeword(const std::vector<std::string>& imlist, cv::OutputArray _code_word);
void decode(cv::InputArray _code_word, std::vector<float>& coor, cv::InputArray _mask);


//cv::Mat grayToDec(const std::vector<cv::Mat1b>&);
//void codeword(const std::vector<std::string>&, std::vector<cv::Mat1b>&);
//void decode(const std::vector<cv::Mat1b>&, std::vector<float>&, cv::InputArray);