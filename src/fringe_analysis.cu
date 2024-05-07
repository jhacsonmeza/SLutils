#include <SLutils/fringe_analysis.hpp>

#include <opencv2/cudaarithm.hpp>


namespace sl {

__global__ void N_phase(const cv::cuda::PtrStepSz<double> sumIcos, const cv::cuda::PtrStep<double> sumIsin,
                        cv::cuda::PtrStep<double> phase) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= sumIcos.rows || j >= sumIcos.cols) return;

    phase(i,j) = -atan2(sumIsin(i,j), sumIcos(i,j));
}

__global__ void N_modulation(const cv::cuda::PtrStepSz<double> sumI,
                             const cv::cuda::PtrStep<double> sumIcos,
                             const cv::cuda::PtrStep<double> sumIsin,
                             cv::cuda::PtrStep<double> data_modulation) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= sumI.rows || j >= sumI.cols) return;
    
    double a1 = sumIcos(i,j);
    double a2 = sumIsin(i,j);
    
    // Estimate numerator: sqrt(sumIcos^2 + sumIsin^2)
    double numerator = hypot(a1, a2); // sqrt(a1*a1 + a2*a2);
    // Final data modulation: sqrt(sumIcos^2 + sumIsin^2)/sumI
    data_modulation(i,j) = numerator/sumI(i,j);
}

__global__ void three_phase(const cv::cuda::PtrStepSzb im1, const cv::cuda::PtrStepb im2,
                            const cv::cuda::PtrStepb im3, cv::cuda::PtrStep<double> phi) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= im1.rows || j >= im1.cols) return;
    
    double I1 = static_cast<double>(im1(i,j));
    double I2 = static_cast<double>(im2(i,j));
    double I3 = static_cast<double>(im3(i,j));
    
    double y = sqrt(3.)*(I1 - I3);
    double x = 2*I2 - I1 - I3;
    
    phi(i,j) = atan2(y, x);
}

__global__ void three_phase_modulation(const cv::cuda::PtrStepSzb im1,
                                       const cv::cuda::PtrStepb im2,
                                       const cv::cuda::PtrStepb im3,
                                       cv::cuda::PtrStep<double> phi, cv::cuda::PtrStep<double> gamma) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= im1.rows || j >= im1.cols) return;
    
    double I1 = static_cast<double>(im1(i,j));
    double I2 = static_cast<double>(im2(i,j));
    double I3 = static_cast<double>(im3(i,j));
    
    double y = sqrt(3.)*(I1 - I3);
    double x = 2*I2 - I1 - I3;
    
    // Phase map
    phi(i,j) = atan2(y, x);
    
    // Data modulation
    double numerator = hypot(x, y);
    gamma(i,j) = numerator/(I1 + I2 + I3);
}


void NStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase, int N) {
    cv::cuda::Stream stream0;

    // Initialize sumIsin and sumIcos with the first fringe image
    cv::Mat I_h = cv::imread(imgs[0], 0);
    I_h.convertTo(I_h, CV_64F);
    cv::cuda::GpuMat I(I_h);
    double delta = 2*CV_PI/N; // delta for i = 0
    
    cv::cuda::GpuMat sumIsin;
    cv::cuda::multiply(I, std::sin(delta), sumIsin, 1, -1, stream0);
    
    cv::cuda::GpuMat sumIcos;
    cv::cuda::multiply(I, std::cos(delta), sumIcos, 1, -1, stream0);
    
    
    // Add the other fringes to sumIsin and sumIcos
    for (int i = 1; i < imgs.size(); i++) {
        cv::Mat I_h = cv::imread(imgs[i], 0);
        I_h.convertTo(I_h, CV_64F);
        cv::cuda::GpuMat I(I_h);
        double delta = 2*CV_PI*(i + 1)/N;

        cv::cuda::scaleAdd(I, std::sin(delta), sumIsin, sumIsin, stream0); // sumIsin += I*std::sin(delta);
        cv::cuda::scaleAdd(I, std::cos(delta), sumIcos, sumIcos, stream0); // sumIcos += I*std::cos(delta);
    }
    
    // Set output wrapped phase array
    _phase.create(sumIsin.size(), sumIsin.type());
    cv::cuda::GpuMat phase = _phase.getGpuMat();
    
    // Estimate final wrapped phase with atan2
    dim3 block(16, 16);
    dim3 grid((phase.cols + block.x - 1)/block.x, (phase.rows + block.y - 1)/block.y);
    N_phase<<<grid, block>>>(sumIcos, sumIsin, phase);
}

void NStepPhaseShifting_modulation(const std::vector<std::string>& imgs, cv::OutputArray _phase,
                                   cv::OutputArray _data_modulation, int N) {
    cv::cuda::Stream stream0;

    // Initialize sumI, sumIsin, and sumIcos using the first fringe image
    cv::Mat sumI_h = cv::imread(imgs[0], 0);
    sumI_h.convertTo(sumI_h, CV_64F);
    cv::cuda::GpuMat sumI(sumI_h);
    double delta = 2*CV_PI/N; // delta for i = 0
    
    cv::cuda::GpuMat sumIsin;
    cv::cuda::multiply(sumI, std::sin(delta), sumIsin, 1, -1, stream0);
    
    cv::cuda::GpuMat sumIcos;
    cv::cuda::multiply(sumI, std::cos(delta), sumIcos, 1, -1, stream0);
    
    
    // Add the other fringes to sumI, sumIsin, and sumIcos
    for (int i = 1; i < imgs.size(); i++) {
        cv::Mat I_h = cv::imread(imgs[i], 0);
        I_h.convertTo(I_h, CV_64F);
        cv::cuda::GpuMat I(I_h);
        double delta = 2*CV_PI*(i + 1)/N;
        
        cv::cuda::add(sumI, I, sumI, {}, -1, stream0); // sumI += I;
        cv::cuda::scaleAdd(I, std::sin(delta), sumIsin, sumIsin, stream0); // sumIsin += I*std::sin(delta);
        cv::cuda::scaleAdd(I, std::cos(delta), sumIcos, sumIcos, stream0); // sumIcos += I*std::cos(delta);
    }
    
    // ------------- Estimate final wrapped phase with atan2
    _phase.create(sumIsin.size(), sumIsin.type());
    cv::cuda::GpuMat phase = _phase.getGpuMat();
    dim3 block(16, 16);
    dim3 grid((phase.cols + block.x - 1)/block.x, (phase.rows + block.y - 1)/block.y);
    N_phase<<<grid, block>>>(sumIcos, sumIsin, phase);
    
    
    // ----------- Estimate data modulation: sqrt(sumIcos^2 + sumIsin^2)/sumI
    cv::cuda::GpuMat numerator;
    cv::cuda::sqr(sumIcos, sumIcos, stream0); // sumIcos^2
    cv::cuda::sqr(sumIsin, sumIsin, stream0); // sumIsin^2
    cv::cuda::add(sumIcos, sumIsin, numerator, {}, -1, stream0); // sumIcos^2 + sumIsin^2
    cv::cuda::sqrt(numerator, numerator, stream0); // sqrt(sumIcos^2 + sumIsin^2)
    cv::cuda::divide(numerator, sumI, _data_modulation, 1, -1, stream0);
}

void ThreeStepPhaseShifting(const std::vector<std::string>& imgs, cv::OutputArray _phase) {
    cv::cuda::Stream stream0;
    
    // Read the three fringe images
    cv::cuda::GpuMat im1, im2, im3;
    
    cv::Mat im1_h = cv::imread(imgs[0], 0);
    im1.upload(im1_h, stream0);
    
    cv::Mat im2_h = cv::imread(imgs[1], 0);
    im2.upload(im2_h, stream0);
    
    cv::Mat im3_h = cv::imread(imgs[2], 0);
    im3.upload(im3_h, stream0);
    
    
    // Set output wrapped phase array
    _phase.create(im1.size(), CV_64F);
    cv::cuda::GpuMat phase = _phase.getGpuMat();
    
    // Estimate final wrapped phase with atan2
    dim3 block(16, 16);
    dim3 grid((phase.cols + block.x - 1)/block.x, (phase.rows + block.y - 1)/block.y);
    three_phase<<<grid, block>>>(im1, im2, im3, phase);
}

void ThreeStepPhaseShifting_modulation(const std::vector<std::string>& imgs, cv::OutputArray _phase,
                                       cv::OutputArray _data_modulation) {
    cv::cuda::Stream stream0;
    
    // Read the three fringe images
    cv::cuda::GpuMat im1, im2, im3;
    
    cv::Mat im1_h = cv::imread(imgs[0], 0);
    im1.upload(im1_h, stream0);
    
    cv::Mat im2_h = cv::imread(imgs[1], 0);
    im2.upload(im2_h, stream0);
    
    cv::Mat im3_h = cv::imread(imgs[2], 0);
    im3.upload(im3_h, stream0);
    
    
    // Set output wrapped phase array
    _phase.create(im1.size(), CV_64F);
    cv::cuda::GpuMat phase = _phase.getGpuMat();
    
    // Set output data modulation array
    _data_modulation.create(phase.size(), phase.type());
    cv::cuda::GpuMat data_modulation = _data_modulation.getGpuMat();
    
    // Estimate final wrapped phase and data modulation arrays
    dim3 block(16, 16);
    dim3 grid((phase.cols + block.x - 1)/block.x, (phase.rows + block.y - 1)/block.y);
    three_phase_modulation<<<grid, block>>>(im1, im2, im3, phase, data_modulation);
}

} // namespace sl
