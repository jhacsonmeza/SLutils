#include <SLutils/graycoding.hpp>

#include <stdexcept> // std::runtime_error


namespace sl {

void decimalMap(const std::vector<std::string>& imlist, cv::OutputArray _dec) {
    if (imlist.size() % 2 != 0)
        throw std::runtime_error("decimalMap requires an even set of images\n");
    
    // Total number of graycode bits (pairs of captured graycode patterns)
    int n = imlist.size()/2;
    
    /* -----------------------------------------------------------------------
    Initialize decimal array (phase order map) 
    using the first pair of graycode images
    ----------------------------------------------------------------------- */
    cv::Mat im1 = cv::imread(imlist[0], 0);
    cv::Mat im2 = cv::imread(imlist[1], 0);
    cv::Mat gray = (im1 > im2)/255;
    
    // Create output array that stores graycode words converted to decimal
    _dec.create(gray.size(), CV_32S);
    cv::Mat dec = _dec.getMat();
    
    // Initialize decimal going from gray to bininary to decimal
    // First binary value (MSB) is the same from gray
    uchar* pgray = gray.data;
    int* pdec = dec.ptr<int>();
    for (size_t i = 0; i < gray.total(); i++)
        pdec[i] = pgray[i] ? 1 << (n - 1) : 0;
    
    
    /* -----------------------------------------------------------------------
    Initializing the binary map, which is equal to the graycode map
    because the Most Significant Bit (MSB) of the binary code = gray code MSB
    -------------------------------------------------------------------------- */
    cv::Mat bin = gray; // this not copy data but increases refcount
    uchar* pbin = bin.data;
    
    
    /* -----------------------------------------------------------------------
    Adding the rest of graycode patterns to estimate the final phase order
    -------------------------------------------------------------------------- */
    for (int k = 1; k < n; k++) {
        // Read graycoding pattern and its inverted counterpart
        cv::Mat im1 = cv::imread(imlist[2*k], 0);
        cv::Mat im2 = cv::imread(imlist[2*k+1], 0);
        // Generate a single gray map
        cv::Mat gray = (im1 > im2) / 255;
        uchar* pgray = gray.data;

        for (size_t i = 0; i < gray.total(); i++) {
            // Convert current gray code bit to binary bit using xor between 
            // the previous binary bit and the current gray bit
            // see: https://www.geeksforgeeks.org/gray-to-binary-and-binary-to-gray-conversion/
            pbin[i] ^= pgray[i];
            
            // if binary bit is 1 then add 2^(bit_pos) to the decimal array
            if (pbin[i]) pdec[i] += 1 << (n - k - 1);
        }
    }
}

void graycodeword(const std::vector<std::string>& imlist, cv::OutputArray _code_word) {
    if (imlist.size() % 2 != 0)
        throw std::runtime_error("graycodeword requires an even set of images\n");
    
    // Total number of graycode bits (pairs of captured graycode patterns)
    int n = imlist.size()/2;

    // Read first image to estimate output array size
    cv::Size sz = cv::imread(imlist[0], 0).size();

    // Setting output 3D array as (n,h,w) array with n graycode patterns of (h,w) size
    int w = sz.width, h = sz.height;
    int dims[] = {n, h, w};
    _code_word.create(3, dims, CV_8U);
    cv::Mat code_word = _code_word.getMat();

    // Estimating gray maps and adding them to the code_word 3D array
    uchar* pcode_word = code_word.data;
    for (int k = 0; k < n; k++) {
        cv::Mat im1 = cv::imread(imlist[2*k], 0);
        cv::Mat im2 = cv::imread(imlist[2*k+1], 0);
        
        cv::Mat bin = (im1 > im2) / 255;
        uchar* pbin = bin.data;
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                pcode_word[k*w*h + i*w + j] = pbin[i*w + j];
    }
}

void gray2dec(cv::InputArray _code_word, cv::OutputArray _dec) {
    cv::Mat code_word = _code_word.getMat();
    int n = code_word.size[0], h = code_word.size[1], w = code_word.size[2];

    // Output array (init with zeros) to store graycode words converted to decimal
    _dec.create(h, w, CV_32S);
    cv::Mat dec = _dec.getMat();
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
}

void decode(cv::InputArray _code_word, std::vector<float>& coor, cv::InputArray _mask) {
    cv::Mat dec;
    gray2dec(_code_word, dec); // int
    cv::Mat mask = _mask.getMat();
    coor.reserve(cv::countNonZero(mask));

    uchar* pmask = mask.data;
    int* pdec = dec.ptr<int>();
    for (int i = 0; i < mask.total(); i++)
        if (pmask[i])
            coor.push_back(static_cast<float>(pdec[i]));
}

} // namespace sl
