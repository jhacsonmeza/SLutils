#include <SLutils/fringe_analysis.hpp>

#include <cmath> // std::atan2, std::sqrt
#include <stdexcept> // std::runtime_error


namespace sl {

void NStepPhaseShifting(const std::vector<std::string>& impaths, cv::OutputArray _phase, int N) {
    if (impaths.size() < 3)
        throw std::runtime_error("NStepPhaseShifting needs at least 3 fringe patterns");

    // Initialize sumIsin and sumIcos with the first fringe image
    cv::Mat I = cv::imread(impaths[0], 0);
    I.convertTo(I, CV_64F); // convert image from uint8 to floating point
    double delta = 2*CV_PI/N; // delta for i = 0
    cv::Mat sumIsin = I*std::sin(delta);
    cv::Mat sumIcos = I*std::cos(delta);
    
    // Add the other fringes to sumIsin and sumIcos
    for (std::size_t i = 1; i < impaths.size(); i++) {
        cv::Mat I = cv::imread(impaths[i], 0);
        I.convertTo(I, CV_64F);
        double delta = 2*CV_PI*(i + 1)/N;
        sumIsin += I*std::sin(delta);
        sumIcos += I*std::cos(delta);
    }
    
    // Set output wrapped phase array
    _phase.create(sumIsin.size(), sumIsin.type());
    cv::Mat phase = _phase.getMat();
    
    // Estimate final wrapped phase with atan2
    double* pphase = phase.ptr<double>();
    double* psumIsin = sumIsin.ptr<double>();
    double* psumIcos = sumIcos.ptr<double>();
    for (std::size_t i = 0; i < sumIsin.total(); i++)
        pphase[i] = -std::atan2(psumIsin[i], psumIcos[i]);
}

void NStepPhaseShifting_modulation(const std::vector<std::string>& impaths, cv::OutputArray _phase,
                                   cv::OutputArray _data_modulation, int N) {
    if (impaths.size() < 3)
        throw std::runtime_error("NStepPhaseShifting_modulation needs at least 3 fringe patterns");
    
    // Initialize sumI, sumIsin, and sumIcos using the first fringe image
    cv::Mat sumI = cv::imread(impaths[0], 0); // In this case sumI = I_0
    sumI.convertTo(sumI, CV_64F); // convert image from uint8 to floating point
    double delta = 2*CV_PI/N; // delta for i = 0
    
    cv::Mat sumIsin = sumI*std::sin(delta);
    cv::Mat sumIcos = sumI*std::cos(delta);
    
    
    // Add the other fringes to sumI, sumIsin, and sumIcos
    for (std::size_t i = 1; i < impaths.size(); i++) {
        cv::Mat I = cv::imread(impaths[i], 0);
        I.convertTo(I, CV_64F);
        double delta = 2*CV_PI*(i + 1)/N;
        
        sumI += I;
        sumIsin += I*std::sin(delta);
        sumIcos += I*std::cos(delta);
    }
    
    // ------------- Estimate final wrapped phase with atan2
    _phase.create(sumIsin.size(), sumIsin.type());
    cv::Mat phase = _phase.getMat();
    double* pphase = phase.ptr<double>();
    double* psumIsin = sumIsin.ptr<double>();
    double* psumIcos = sumIcos.ptr<double>();
    for (std::size_t i = 0; i < sumIsin.total(); i++)
        pphase[i] = -std::atan2(psumIsin[i], psumIcos[i]);
    
    // ----------- Estimate data modulation: sqrt(sumIcos^2 + sumIsin^2)/sumI
    cv::Mat numerator = sumIcos.mul(sumIcos) + sumIsin.mul(sumIsin);
    cv::sqrt(numerator, numerator);
    cv::Mat data_modulation = numerator/sumI;
    _data_modulation.assign(data_modulation);
}

void ThreeStepPhaseShifting(const std::vector<std::string>& impaths, cv::OutputArray _phase) {
    if (impaths.size() != 3)
        throw std::runtime_error("ThreeStepPhaseShifting needs exactly 3 fringe patterns");
    
    // Read the three fringe images
    cv::Mat im1 = cv::imread(impaths[0], 0);
    cv::Mat im2 = cv::imread(impaths[1], 0);
    cv::Mat im3 = cv::imread(impaths[2], 0);
    
    // Set output wrapped phase array
    _phase.create(im1.size(), CV_64F);
    cv::Mat phase = _phase.getMat();
    
    // Estimate final wrapped phase with atan2
    double* pphase = phase.ptr<double>();
    uchar *pim1 = im1.data, *pim2 = im2.data, *pim3 = im3.data;
    for (std::size_t i = 0; i < phase.total(); i++) {
        double I1 = static_cast<double>(pim1[i]);
        double I2 = static_cast<double>(pim2[i]);
        double I3 = static_cast<double>(pim3[i]);
        
        pphase[i] = std::atan2(std::sqrt(3.)*(I1 - I3), 2*I2 - I1 - I3);
    }
}

void ThreeStepPhaseShifting_modulation(const std::vector<std::string>& impaths, cv::OutputArray _phase,
                                       cv::OutputArray _data_modulation) {
    if (impaths.size() != 3)
        throw std::runtime_error("ThreeStepPhaseShifting_modulation needs exactly 3 fringe patterns");
    
    // Read the three fringe images
    cv::Mat im1 = cv::imread(impaths[0], 0);
    cv::Mat im2 = cv::imread(impaths[1], 0);
    cv::Mat im3 = cv::imread(impaths[2], 0);
    
    // Set output wrapped phase array
    _phase.create(im1.size(), CV_64F);
    cv::Mat phase = _phase.getMat();
    
    // Set output data modulation array
    _data_modulation.create(phase.size(), phase.type());
    cv::Mat data_modulation = _data_modulation.getMat();
    
    
    // Estimate final wrapped phase and data modulation arrays
    double* pphase = phase.ptr<double>();
    double* gamma = data_modulation.ptr<double>();
    uchar *pim1 = im1.data, *pim2 = im2.data, *pim3 = im3.data;
    for (std::size_t i = 0; i < phase.total(); i++) {
        double I1 = static_cast<double>(pim1[i]);
        double I2 = static_cast<double>(pim2[i]);
        double I3 = static_cast<double>(pim3[i]);
        
        double num = std::sqrt(3.)*(I1 - I3);
        double den = 2*I2 - I1 - I3;
        
        // Phase map
        pphase[i] = std::atan2(num, den);
        
        // Data modulation
        gamma[i] = std::sqrt(num*num + den*den)/(I1 + I2 + I3);
    }
}

} // namespace sl
