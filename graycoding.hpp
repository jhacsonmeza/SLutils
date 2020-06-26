#ifndef GRAYCODING_HPP
#define GRAYCODING_HPP

#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>

cv::Mat1i grayToDec(cv::InputArray);
void codeword(const std::vector<std::string>&, cv::OutputArray);
void decode(cv::InputArray, std::vector<float>&, cv::InputArray);


//cv::Mat1i grayToDec(const std::vector<cv::Mat1b>&);
//void codeword(const std::vector<std::string>&, std::vector<cv::Mat1b>&);
//void decode(const std::vector<cv::Mat1b>&, std::vector<float>&, cv::InputArray);

#endif