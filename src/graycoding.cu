#include <SLutils/graycoding.hpp>

#include <opencv2/core/cuda.hpp>
#include <stdexcept> // std::runtime_error


namespace sl {

__global__ void initDecimalArray(const cv::cuda::PtrStepSzb gray, cv::cuda::PtrStepi decimal, int n_bits) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= gray.rows || j >= gray.cols) return;

    decimal(i,j) = gray(i,j) ? 1 << (n_bits - 1) : 0;
}

__global__ void gray2dec_array(const cv::cuda::PtrStepSzb gray, cv::cuda::PtrStepb bin,
                               cv::cuda::PtrStepi decimal, int n_bits, int pos) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= gray.rows || j >= gray.cols) return;

    // Convert current gray code bit to binary bit using xor between 
    // the previous binary bit and the current gray bit
    // see: https://www.geeksforgeeks.org/gray-to-binary-and-binary-to-gray-conversion/
    bin(i,j) ^= gray(i,j);
    // if binary bit is 1 then add 2^(bit_pos) to the decimal array
    if (bin(i,j)) decimal(i,j) += 1 << (n_bits - pos - 1);
}


void decimalMap(const std::vector<std::string>& impaths, cv::OutputArray _dec) {
    if (impaths.size() > 1 and impaths.size() % 2 != 0)
        throw std::runtime_error("decimalMap requires an even set of images");

    cv::cuda::Stream stream0;
    
    // Total number of graycode bits (pairs of captured graycode patterns)
    int n = impaths.size()/2;
    

    /* -----------------------------------------------------------------------
    Initialize decimal array (phase order map) using the first pair of
    graycode images. Also the binary map, which is equal to the graycode map
    because the Most Significant Bit (MSB) of the binary code = MSB gray code
    ----------------------------------------------------------------------- */
    cv::Mat im1_h = cv::imread(impaths[0], 0);
    cv::cuda::GpuMat im1;
    im1.upload(im1_h, stream0);
    
    cv::Mat im2_h = cv::imread(impaths[1], 0);
    cv::cuda::GpuMat im2;
    im2.upload(im2_h, stream0);

    // Allocate output decimal array which is obtained from graycode words
    _dec.create(im1_h.size(), CV_32S);
    cv::cuda::GpuMat dec = _dec.getGpuMat();
    
    // Allocate binary array
    cv::cuda::GpuMat bin(im1_h.size(), CV_8U);

    // Launching initDecimalAndBinary to initialize the values of dec and bin
    dim3 block(16, 16);
    dim3 grid((dec.cols + block.x - 1)/block.x, (dec.rows + block.y - 1)/block.y);
    initDecimalAndBinary<<<grid, block>>>(im1, im2, dec, bin, n);


    /* -----------------------------------------------------------------------
    Adding the rest of graycode patterns to estimate the final phase order
    -------------------------------------------------------------------------- */
    for (int i = 1; i < n; i++) {
        // Read graycoding pattern and its inverted counterpart
        cv::Mat im1_h = cv::imread(impaths[2*i], 0);
        cv::cuda::GpuMat im1;
        im1.upload(im1_h, stream0);
        
        cv::Mat im2_h = cv::imread(impaths[2*i+1], 0);
        cv::cuda::GpuMat im2;
        im2.upload(im2_h, stream0);

        dec_array<<<grid, block>>>(im1, im2, bin, dec, n, i);
    }
}

void graycodeword(const std::vector<std::string>& impaths, cv::OutputArray _code_word) {
    if (impaths.size() > 1 and impaths.size() % 2 != 0)
        throw std::runtime_error("graycodeword requires an even set of images");

    cv::cuda::Stream stream0;
    
    // Total number of graycode bits (pairs of captured graycode patterns)
    int n = impaths.size()/2;

    // Read first image to obtain the output array size
    cv::Size sz = cv::imread(impaths[0], 0).size();

    // Get output vector of arrays
    std::vector<cv::cuda::GpuMat>& gray_images = _code_word.getGpuMatVecRef();

    for (int k = 0; k < n; k++) {
        // Read graycoding pattern and its inverted counterpart
        cv::Mat im1 = cv::imread(impaths[2*k], 0);
        cv::Mat im2 = cv::imread(impaths[2*k+1], 0);
        // Generate a single gray map
        cv::Mat gray_h = (im1 > im2)/255;

        // Convert to GPU with continuous memory block of byte data
        cv::cuda::GpuMat gray;
        gray.upload(gray_h, stream0);
        gray_images.push_back(gray);
    }
}

void gray2dec(cv::InputArray _code_word, cv::OutputArray _dec) {
    // Obtain input vector of GpuMats
    std::vector<cv::cuda::GpuMat> code_word;
    _code_word.getGpuMatVector(code_word);
    if (code_word.size() < 2)
        throw std::runtime_error("gray2dec needs at least more than 1 graycode map");

    // Number of graycoding arrays, rows, and columns of the images
    int n = code_word.size(), h = code_word[0].rows, w = code_word[0].cols;


    // Output array that store graycode words converted to decimal
    _dec.create(h, w, CV_32S);
    cv::cuda::GpuMat dec = _dec.getGpuMat();

    // Launching initDecimalArray to initialize the values of dec
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1)/block.x, (h + block.y - 1)/block.y);
    initDecimalArray<<<grid, block>>>(code_word[0], dec, n);

    // Initializing the binary map. 
    // Where the Most Significant Bit (MSB) of the binary code = MSB gray code
    cv::cuda::GpuMat bin = code_word[0].clone();
    // Convert from gray code to decimal
    for (int i = 1; i < n; i++)
        gray2dec_array<<<grid, block>>>(code_word[i], bin, dec, n, i);
}

} // namespace sl
