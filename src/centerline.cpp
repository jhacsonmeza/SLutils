#include <phase_unwrap/centerline.hpp>

#include <opencv2/imgcodecs.hpp> // cv::imread
#include <opencv2/imgproc.hpp> // cv::threshold

#include <queue>


namespace sl {

cv::Point seedPoint(const std::string& fn_clx, const std::string& fn_cly, cv::InputArray _mask) {
    // Read center line images
    cv::Mat clx = cv::imread(fn_clx, 0);
    cv::Mat cly = cv::imread(fn_cly, 0);
    // Get input mask
    cv::Mat mask = _mask.getMat();
    
    // Estimate vertical line
    cv::bitwise_and(clx, mask, clx);
    cv::Mat bw1;
    cv::threshold(clx, bw1, 0, 255, cv::THRESH_OTSU+cv::THRESH_BINARY);
    
    // Estimate horizontal line
    cv::bitwise_and(cly, mask, cly);
    cv::Mat bw2;
    cv::threshold(cly, bw2, 0, 255, cv::THRESH_OTSU+cv::THRESH_BINARY);
    
    // Estimate centroid of the intersection between both binary lines
    float sum_x = 0, sum_y = 0;
    int count{0};
    
    int h = mask.rows, w = mask.cols;
    uchar* pbw1 = bw1.data;
    uchar* pbw2 = bw2.data;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (pbw1[i*w + j] && pbw2[i*w + j]) {
                sum_x += j;
                sum_y += i;
                count++;
            }
        }
    }
    
    // Estimate mean
    int x = cvRound(sum_x/count);
    int y = cvRound(sum_y/count);
    
    return {x, y};
}

void spatialUnwrap(cv::InputArray _phased, const cv::Point p0, cv::InputArray _mask, cv::OutputArray _Phi) {
    // Define offsets
    constexpr int xo[8] = {-1, 0, 1,-1, 1,-1, 0, 1};
    constexpr int yo[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
    
    // Get input discontinuous phase map
    cv::Mat phased = _phased.getMat();
    // Get a editable input mask
    cv::Mat mask;
    _mask.copyTo(mask);
    
    // Initialize output continuous phase map
    cv::Mat phasec = phased.clone();
    
    // Initialize a queue to store the unwrapped points
    std::queue<cv::Point> queue;
    queue.push(p0);  // The first point is p0
    
    
    const double* pphased = phased.ptr<double>();
    double* pphasec = phasec.ptr<double>();
    uchar* pmask = mask.data;
    
    // Remove p0 from the mask
    const int h = phased.rows, w = phased.cols;
    pmask[p0.y*w + p0.x] = 0;
    
    while (!queue.empty()) {
        // Get first element from the queue (current point)
        cv::Point p = queue.front();
        // Remove that element (visited)
        queue.pop();
        
        // Get continuous and discontinuous phase values in p
        const double PCI = pphasec[p.y*w + p.x];
        const double PDI = pphased[p.y*w + p.x];
        
        // Unwrap the 8-neighbors of p
        for (int i = 0; i < 8; i++) {
            const int px = p.x + xo[i];
            const int py = p.y + yo[i];
            
            // Check if point is outisde the mask and image bounds to ignore it
            if (py < 0 || py >= h || px < 0 || px >= w || !pmask[py*w + px]) continue;
            
            // Get wrapped phase value at the p's neighbor
            const double PDC = pphased[py*w + px];
            
            // Unwrap p's neighbor
            double D = (PDC - PDI)/(2*CV_PI);
            pphasec[py*w + px] = PCI + 2*CV_PI*(D - cvRound(D));
            
            // Add the unwrapped point to the queue
            queue.emplace(px, py);
            
            // Remove unwrapped point from the mask
            pmask[py*w + px] = 0;
        }
    }
    
    _Phi.assign(phasec);
}

} // namespace sl
