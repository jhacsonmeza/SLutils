#include <SLutils/phase_graycoding.hpp>

#include <SLutils/fringe_analysis.hpp> // NStepPhaseShifting
#include <SLutils/graycoding.hpp> // decimalMap

#include <opencv2/core/cuda.hpp>


namespace sl {

__global__ void unwrapWithPhaseOrder(const cv::cuda::PtrStepSz<double> phi, const cv::cuda::PtrStepi k,
                                     cv::cuda::PtrStep<double> Phi, double shift) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= phi.rows || j >= phi.cols) return;

    // Shift and rewrap wrapped phase values
    double phi_ij = phi(i,j) + shift;
    double phi_shifted = atan2(sin(phi_ij), cos(phi_ij));

    // Estimate unwrapped phase map with k order map: Phi = phi + 2*pi*k
    double Phi_shifted = phi_shifted + 2*CV_PI*k(i,j);

    // Shift phase back to the original values
    Phi(i,j) = Phi_shifted - shift;
}

__global__ void removeSpikyNoise(cv::cuda::PtrStepSz<double> Phi) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    constexpr int ksize{5};
    int mid = ksize/2;
    if (i < mid || i > Phi.rows-1-mid || j < mid || j > Phi.cols-1-mid) return;

    // -------------------------- Median filter
    double values[ksize*ksize];
    for (int row = 0; row < ksize; row++)
        for (int col = 0; col < ksize; col++)
        {
            int m = row*ksize + col;
            values[m] = Phi(i+row-mid, j+col-mid);

            // Sorting the elements (Insertion Sort)
            if (m != 0)
            {
                double v = values[m];
                int n = m - 1;
                while (n >= 0 && values[n] > v)
                {
                    values[n+1] = values[n];
                    n--;
                }
                values[n+1] = v;
            }
        }

    // Get the median phase value at (i,j)
    double Phim = values[ksize*ksize/2];

    // -------------------------- Remove spiky points
    // Determine order of 2*pi to add to remove spiky points
    Phi(i,j) -= 2*CV_PI*round( (Phi(i,j) - Phim)/2/CV_PI );
}

void phaseGraycodingUnwrap(const std::vector<std::string>& impaths_ps,
                           const std::vector<std::string>& impaths_gc,
                           cv::OutputArray _Phi, int p, int N) {
    // Estimate wrapped phase map
    cv::cuda::GpuMat phi; // double mat
    NStepPhaseShifting(impaths_ps, phi, N);
    
    // Estimate decimal map (phase order) with the gray patterns
    cv::cuda::GpuMat k;
    decimalMap(impaths_gc, k);


    // --- Phase unwrapping using the phase order map k
    dim3 block(16, 16);
    dim3 grid((phi.cols + block.x - 1)/block.x, (phi.rows + block.y - 1)/block.y);
    double shift = -CV_PI + CV_PI/p;
    // Get output array
    _Phi.create(phi.size(), phi.type());
    cv::cuda::GpuMat Phi = _Phi.getGpuMat();
    // Launch kernel
    unwrapWithPhaseOrder<<<grid, block>>>(phi, k, Phi, shift);


    // --- Remove spiky noise using median filter
    removeSpikyNoise<<<grid, block>>>(Phi);
}

} // namespace sl
