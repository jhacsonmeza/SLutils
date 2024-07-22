#include <SLutils/multifrequency.hpp>

#include <SLutils/fringe_analysis.hpp> // NStepPhaseShifting

#include <opencv2/core/cuda.hpp>


namespace sl {

__global__ void equivalentPhase(const cv::cuda::PtrStepSz<double> phase1,
                                const cv::cuda::PtrStep<double> phase2,
                                cv::cuda::PtrStep<double> eqPhase) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= phase1.rows || j >= phase1.cols) return;
    
    constexpr double twoPI = 2*CV_PI;
    
    // Phase difference
    double diff = phase1(i,j) - phase2(i,j);
    
    // Estimate modulus
    double mod = fmodf(diff, twoPI);
    if (mod < 0) mod += twoPI;

    eqPhase(i,j) = mod;
}

__global__ void backwardUnwrap(const cv::cuda::PtrStepSz<double> phase1,
                               cv::cuda::PtrStep<double> phase2, double T1, double T2) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= phase1.rows || j >= phase1.cols) return;
    
    constexpr double twoPI = 2*CV_PI;
    
    double phi2 = phase2(i,j);
    
    // Estimate phase order
    double k = (T1/T2*phase1(i,j) - phi2)/twoPI;
    
    // Unwrap phase value
    phase2(i,j) = phi2 + twoPI*round(k);
}

__global__ void removeSpikyNoise(cv::cuda::PtrStepSz<double> Phi) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    constexpr int ksize{5};
    int mid = ksize/2;
    if (i < mid || i > Phi.rows-1-mid || j < mid || j > Phi.cols-1-mid) return;

    // -------------------------- Median filter
    double values[ksize*ksize];
    for (int row = 0; row < ksize; row++) {
        for (int col = 0; col < ksize; col++) {
            int m = row*ksize + col;
            values[m] = Phi(i+row-mid, j+col-mid);

            // Sorting the elements (Insertion Sort)
            if (m != 0) {
                double v = values[m];
                int n = m - 1;
                while (n >= 0 && values[n] > v) {
                    values[n+1] = values[n];
                    n--;
                }
                values[n+1] = v;
            }
        }
    }

    // Get the median phase value at (i,j)
    double Phim = values[ksize*ksize/2];

    // -------------------------- Remove spiky points
    // Determine order of 2*pi to add to remove spiky points
    Phi(i,j) -= 2*CV_PI*round( (Phi(i,j) - Phim)/2/CV_PI );
}

void threeFreqPhaseUnwrap(const std::vector<std::string>& impaths, cv::OutputArray _Phi,
                          const cv::Vec3i& p, const cv::Vec3i& N) {
    if (impaths.size() != (N[0]+N[1]+N[2]))
        throw std::runtime_error("threeFreqPhaseUnwrap: number of image paths and number of patterns N must match.");
    
    // Get input fringe periods
    double T1 = p[0], T2 = p[1], T3 = p[2];
    // Estimate equivalent intermidate periods
    double T12 = T1*T2/std::abs(T1-T2);
    double T23 = T2*T3/std::abs(T2-T3);
    double T123 = T12*T3/std::abs(T12-T3);
    
    // ------------- Estimating wrapped phase map for each frequency
    cv::cuda::GpuMat phi1, phi2, phi3;
    NStepPhaseShifting({impaths.begin(), impaths.begin()+N[0]}, phi1, N[0]);
    NStepPhaseShifting({impaths.begin()+N[0], impaths.begin()+N[0]+N[1]}, phi2, N[1]);
    NStepPhaseShifting({impaths.end()-N[2], impaths.end()}, phi3, N[2]);
    

    // ------------- Estimate equivalent phase maps
    dim3 block(16, 16);
    dim3 grid((phi1.cols + block.x - 1)/block.x, (phi1.rows + block.y - 1)/block.y);
    
    cv::cuda::GpuMat phi12(phi1.size(), phi1.type());
    equivalentPhase<<<grid, block>>>(phi1, phi2, phi12);
    
    cv::cuda::GpuMat phi23(phi1.size(), phi1.type());
    equivalentPhase<<<grid, block>>>(phi2, phi3, phi23);
    
    cv::cuda::GpuMat Phi123(phi1.size(), phi1.type());
    equivalentPhase<<<grid, block>>>(phi12, phi3, Phi123); // Phi123 is a wide phase without discontinuities
    
    
    // ------------- Remove spiky noise in the equivalent phase of wider pitch
    removeSpikyNoise<<<grid, block>>>(Phi123);

    
    // ------------- Backward phase unwrapping
    backwardUnwrap<<<grid, block>>>(Phi123, phi23, T123, T23); // Estimate unwrapped version of phi23
    backwardUnwrap<<<grid, block>>>(phi23, phi12, T23, T12); // Estimate unwrapped version of phi12
    backwardUnwrap<<<grid, block>>>(phi12, phi3, T12, T3); // Estimate unwrapped version of phi3
    backwardUnwrap<<<grid, block>>>(phi3, phi2, T3, T2); // Estimate unwrapped version of phi2
    backwardUnwrap<<<grid, block>>>(phi2, phi1, T2, T1); // Estimate unwrapped version of phi1
    
    phi1.copyTo(_Phi);
}

void twoFreqPhaseUnwrap(const std::vector<std::string>& impaths, cv::OutputArray _Phi,
                        const cv::Vec3i& p, const cv::Vec3i& N) {
    if (impaths.size() != (N[0]+N[1]))
        throw std::runtime_error("twoFreqPhaseUnwrap: number of image paths and number of patterns N must match.");
    
    // Get input fringe periods
    double T1 = p[0], T2 = p[1];
    // Estimate equivalent period
    double T12 = T1*T2/std::abs(T1-T2);
    
    // Estimating wrapped phase map for each frequency
    cv::cuda::GpuMat phi1, phi2;
    NStepPhaseShifting({impaths.begin(), impaths.begin()+N[0]}, phi1, N[0]);
    NStepPhaseShifting({impaths.begin()+N[0], impaths.end()}, phi2, N[1]);
    

    // Estimate equivalent phase map
    dim3 block(16, 16);
    dim3 grid((phi1.cols + block.x - 1)/block.x, (phi1.rows + block.y - 1)/block.y);
    
    cv::cuda::GpuMat Phi12(phi1.size(), phi1.type());
    equivalentPhase<<<grid, block>>>(phi1, phi2, Phi12); // Phi12 is a phase map without discontinuities
    
    
    // Remove spiky noise in the equivalent phase of wider pitch
    removeSpikyNoise<<<grid, block>>>(Phi12);
    
    // Backward phase unwrapping
    backwardUnwrap<<<grid, block>>>(Phi12, phi2, T12, T2); // Estimate unwrapped version of phi2
    backwardUnwrap<<<grid, block>>>(phi2, phi1, T2, T1); // Estimate unwrapped version of phi1
    
    phi1.copyTo(_Phi);
}

} // namespace sl
