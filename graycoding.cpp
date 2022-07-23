#include "graycoding.hpp"

#include <cmath>

cv::Mat1i grayToDec(cv::InputArray _code_word)
{
    cv::Mat code_word = _code_word.getMat();
    int n = code_word.size[0], h = code_word.size[1], w = code_word.size[2];

    // array (init with zeros) to store graycode words converted to decimal
    cv::Mat1i dec(h, w, 0);

    uchar* pcode_word = code_word.data;
    int* pdec = dec.ptr<int>();
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
        {
            int dec_ = 0;
            uchar tmp = pcode_word[i*w + j];
            if (tmp)
                dec_ += 1 << (n - 1);

            for (int k = 1; k < n; k++)
            {
                tmp ^= pcode_word[i*w + j + k*w*h];
                if (tmp)
                    dec_ += 1 << (n - k - 1);
            }

            pdec[i*w + j] = dec_;
        }

    return dec;
}

void codeword(const std::vector<std::string>& imlist, cv::OutputArray _code_word)
{
    // Gruping images as pairs
    std::vector<std::vector<std::string>> l;
    size_t length = imlist.size() / 2;
    size_t begin = 0, end = 0;
    for (int i = 0; i < length; i++)
    {
        end += 2;
        l.push_back(std::vector<std::string>(imlist.begin() + begin, imlist.begin() + end));
        begin = end;
    }

    // Read first image for output array size
    cv::Size sz = cv::imread(l[0][0], 0).size();

    // Setting output 3D array as (n,w,h) array with n graycode patterns of (w,h) size
    int w = sz.width, h = sz.height;
    int dims[] = { static_cast<int>(l.size()), h, w };
    _code_word.create(3, dims, CV_8U);
    cv::Mat code_word = _code_word.getMat();

    uchar* pcode_word = code_word.data;
    for (int k = 0; k < l.size(); k++)
    {
        cv::Mat1b im1 = cv::imread(l[k][0], 0);
        cv::Mat1b im2 = cv::imread(l[k][1], 0);
        cv::Mat1b bin = (im1 > im2) / 255;
        uchar* pbin = bin.data;
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                pcode_word[i*w + j + k*w*h] = pbin[i*w + j];
    }
}

void decode(cv::InputArray _code_word, std::vector<float>& coor, cv::InputArray _mask)
{
    cv::Mat1i dec = grayToDec(_code_word);
    cv::Mat1b mask = _mask.getMat();
    coor.reserve(cv::countNonZero(mask));

    uchar* pmask = mask.data;
    int* pdec = dec.ptr<int>();
    for (int i = 0; i < mask.total(); i++)
        if (pmask[i])
            coor.push_back(static_cast<float>(pdec[i]));
}
