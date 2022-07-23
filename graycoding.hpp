#ifndef GRAYCODING_HPP
#define GRAYCODING_HPP

#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>

cv::Mat1i grayToDec(cv::InputArray _code_word);
void codeword(const std::vector<std::string>& imlist, cv::OutputArray _code_word);
void decode(cv::InputArray _code_word, std::vector<float>& coor, cv::InputArray _mask);


//cv::Mat1i grayToDec(const std::vector<cv::Mat1b>&);
//void codeword(const std::vector<std::string>&, std::vector<cv::Mat1b>&);
//void decode(const std::vector<cv::Mat1b>&, std::vector<float>&, cv::InputArray);

#endif