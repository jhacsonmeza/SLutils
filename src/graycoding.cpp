#include <phase_unwrap/graycoding.hpp>

#include <cmath>


void graycodeword(const std::vector<std::string>& imlist, cv::OutputArray _code_word)
{
    // Gruping images as pairs
    std::vector<std::vector<std::string>> l;
    size_t length = imlist.size() / 2;
    size_t begin = 0, end = 0;
    for (int i = 0; i < length; i++) {
        end += 2;
        l.push_back({imlist.begin() + begin, imlist.begin() + end});
        begin = end;
    }

    // Read first image to estimate output array size
    cv::Size sz = cv::imread(l[0][0], 0).size();

    // Setting output 3D array as (n,h,w) array with n graycode patterns of (h,w) size
    int w = sz.width, h = sz.height;
    int dims[] = {static_cast<int>(l.size()), h, w};
    _code_word.create(3, dims, CV_8U);
    cv::Mat code_word = _code_word.getMat();

    // Estimating gray maps and adding them to the code_word 3D array
    uchar* pcode_word = code_word.data;
    for (int k = 0; k < l.size(); k++) {
        cv::Mat im1 = cv::imread(l[k][0], 0);
        cv::Mat im2 = cv::imread(l[k][1], 0);
        
        cv::Mat bin = (im1 > im2) / 255;
        uchar* pbin = bin.data;
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                pcode_word[k*w*h + i*w + j] = pbin[i*w + j];
    }
}



cv::Mat gray2dec(cv::InputArray _code_word)
{
    cv::Mat code_word = _code_word.getMat();
    int n = code_word.size[0], h = code_word.size[1], w = code_word.size[2];

    // array (init with zeros) to store graycode words converted to decimal
    cv::Mat dec(h, w, CV_32S);
    // array to store graycode bits converted to binary
    cv::Mat bin(h, w, CV_8U);
    
    
    /* -------------------------------------------------------------------------------------------
    // Init decimal and bin arrays knowing that first bin value (MSB) is equal to the gray value
    // We only use pcode_word[i] because this get all the values of the first gray map
    ------------------------------------------------------------------------------------------- */
    int* pdec = dec.ptr<int>();
    uchar* pbin = bin.data;
    uchar* pcode_word = code_word.data;
    for (size_t i = 0; i < dec.total(); i++) {
        uchar graybit = pcode_word[i];
        pdec[i] = graybit ? 1 << (n - 1) : 0;
        pbin[i] = graybit;
    }
    
    /* -----------------------------------------------------------------------
    Adding the rest of graycode patterns to estimate the final phase order
    -------------------------------------------------------------------------- */
    for (int k = 1; k < n; k++) {
        for (size_t i = 0; i < dec.total(); i++) {
            // Convert current gray code bit to binary bit using xor between 
            // the previous binary bit and the current gray bit
            // see: https://www.geeksforgeeks.org/gray-to-binary-and-binary-to-gray-conversion/
            pbin[i] ^= pcode_word[k*w*h + i];
            
            // if binary bit is 1 then add 2^(bit_pos) to the decimal array
            if (pbin[i]) pdec[i] += 1 << (n - k - 1);
        }
    }
    

    return dec;
}

void decode(cv::InputArray _code_word, std::vector<float>& coor, cv::InputArray _mask)
{
    cv::Mat dec = gray2dec(_code_word); // int
    cv::Mat mask = _mask.getMat();
    coor.reserve(cv::countNonZero(mask));

    uchar* pmask = mask.data;
    int* pdec = dec.ptr<int>();
    for (int i = 0; i < mask.total(); i++)
        if (pmask[i])
            coor.push_back(static_cast<float>(pdec[i]));
}
