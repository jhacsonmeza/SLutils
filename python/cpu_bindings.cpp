#include <phase_unwrap/centerline.hpp>
#include <phase_unwrap/fringe_analysis.hpp>
#include <phase_unwrap/graycoding.hpp>
#include <phase_unwrap/phase_graycoding.hpp>

#include <vector>
#include <string>
#include <utility> // std::pair

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;


void delete_Mat(void* p) noexcept {
    delete reinterpret_cast<cv::Mat*>(p);
}


/* ----------------------- Bindings for centerline.hpp ----------------------- */
nb::ndarray<nb::numpy, int> bind_seedPoint(const std::string& fn_clx,
                                           const std::string& fn_cly,
                                           nb::ndarray<uchar, nb::ndim<2>> _mask) {
    
    // Create cv::Mat view
    const size_t h = _mask.shape(0), w = _mask.shape(1);
    cv::Mat mask(h, w, CV_8U, _mask.data());
    
    // Run core function
    cv::Point p0 = sl::seedPoint(fn_clx, fn_cly, mask);
    
    // Dynamically allocate output data
    int* data = new int[2]{p0.x, p0.y};
    
    // Create capsule for the output numpy array
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] static_cast<int*>(p);
    });
    
    return {data, {2}, owner};
}

nb::ndarray<nb::numpy, double> bind_spatialUnwrap(nb::ndarray<double, nb::ndim<2>> _phased,
                                                  nb::ndarray<int, nb::shape<2>> p0,
                                                  nb::ndarray<uchar, nb::ndim<2>> _mask) {
    
    // Get array size
    const size_t h = _phased.shape(0), w = _phased.shape(1);
    
    // Create cv::Mat views
    cv::Mat phased(h, w, CV_64F, _phased.data());
    cv::Mat mask(h, w, CV_8U, _mask.data());
    
    // Run core function
    cv::Mat Phi;
    sl::spatialUnwrap(phased, {p0(0), p0(1)}, mask, Phi);
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::Mat(Phi), delete_Mat);
    
    return {Phi.data, {h, w}, owner};
}


/* ----------------------- Bindings for fringe_analysis.hpp ----------------------- */
nb::ndarray<nb::numpy, double> bind_NStepPhaseShifting(const std::vector<std::string>& imgs, int N) {
    // Run core function
    cv::Mat phi;
    sl::NStepPhaseShifting(imgs, phi, N);
    
    // Get output size
    const size_t h = phi.rows, w = phi.cols;
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::Mat(phi), delete_Mat);
    
    return {phi.data, {h, w}, owner};
}

auto bind_NStepPhaseShifting_modulation(const std::vector<std::string>& imgs, int N)
  -> std::pair<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>> {
    // Run core function
    cv::Mat phi, mod;
    sl::NStepPhaseShifting_modulation(imgs, phi, mod, N);
    
    // Get output size
    const size_t h = phi.rows, w = phi.cols;
    
    // Create capsules for output numpy arrays
    nb::capsule owner_phi(new cv::Mat(phi), delete_Mat);
    nb::capsule owner_mod(new cv::Mat(mod), delete_Mat);
    
    return {{phi.data, {h, w}, owner_phi}, {mod.data, {h, w}, owner_mod}};
}

nb::ndarray<nb::numpy, double> bind_ThreeStepPhaseShifting(const std::vector<std::string>& imgs) {
    // Run core function
    cv::Mat phi;
    sl::ThreeStepPhaseShifting(imgs, phi);
    
    // Get output size
    const size_t h = phi.rows, w = phi.cols;
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::Mat(phi), delete_Mat);
    
    return {phi.data, {h, w}, owner};
}

auto bind_ThreeStepPhaseShifting_modulation(const std::vector<std::string>& imgs)
  -> std::pair<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>> {
    // Run core function
    cv::Mat phi, mod;
    sl::ThreeStepPhaseShifting_modulation(imgs, phi, mod);
    
    // Get output size
    const size_t h = phi.rows, w = phi.cols;
    
    // Create capsules for output numpy arrays
    nb::capsule owner_phi(new cv::Mat(phi), delete_Mat);
    nb::capsule owner_mod(new cv::Mat(mod), delete_Mat);
    
    return {{phi.data, {h, w}, owner_phi}, {mod.data, {h, w}, owner_mod}};
}


/* ----------------------- Bindings for graycoding.hpp ----------------------- */
nb::ndarray<nb::numpy, int> bind_decimalMap(const std::vector<std::string>& imlist) {
    // Run core function
    cv::Mat dec;
    sl::decimalMap(imlist, dec); // returns int 2D array
    
    // Get output size
    const size_t h = dec.rows, w = dec.cols;
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::Mat(dec), delete_Mat);
    
    return {dec.data, {h, w}, owner};
}

nb::ndarray<nb::numpy, uchar> bind_graycodeword(const std::vector<std::string>& imlist) {
    // Run core function
    cv::Mat code_word;
    sl::graycodeword(imlist, code_word); // returns uchar 3D array
    
    // Get output size
    const size_t n = code_word.size[0], h = code_word.size[1], w = code_word.size[2];
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::Mat(code_word), delete_Mat);
    
    return {code_word.data, {n, h, w}, owner};
}

nb::ndarray<nb::numpy, int> bind_gray2dec(nb::ndarray<uchar, nb::ndim<3>> _code_word) {
    // Create cv::Mat view
    const size_t n = _code_word.shape(0), h = _code_word.shape(1), w = _code_word.shape(2);
    int dims[]{static_cast<int>(n), static_cast<int>(h), static_cast<int>(w)};
    cv::Mat code_word(3, dims, CV_8U, _code_word.data());
    
    // Run core function
    cv::Mat dec;
    sl::gray2dec(code_word, dec); // returns int 2D array
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::Mat(dec), delete_Mat);
    
    return {dec.data, {h, w}, owner};
}


/* ----------------------- Bindings for phase_graycoding.hpp ----------------------- */
nb::ndarray<nb::numpy, double> bind_phaseGraycodingUnwrap(const std::vector<std::string>& imlist_ps,
                                                          const std::vector<std::string>& imlist_gc,
                                                          int p, int N) {
    
    // Run core function
	cv::Mat Phi;
	sl::phaseGraycodingUnwrap(imlist_ps, imlist_gc, Phi, p, N);
    
    // Get output size
    const size_t h = Phi.rows, w = Phi.cols;
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::Mat(Phi), delete_Mat);
    
    return {Phi.data, {h, w}, owner};
}



/////////////////////////////////////////////////////////////////////
/* ----------------------- Create bindings ----------------------- */
/////////////////////////////////////////////////////////////////////
NB_MODULE(sl, m) {
    m.def("seedPoint", bind_seedPoint);
    m.def("spatialUnwrap", bind_spatialUnwrap);
    
    m.def("NStepPhaseShifting", bind_NStepPhaseShifting);
    m.def("NStepPhaseShifting_modulation", bind_NStepPhaseShifting_modulation);
    m.def("ThreeStepPhaseShifting", bind_ThreeStepPhaseShifting);
    m.def("ThreeStepPhaseShifting_modulation", bind_ThreeStepPhaseShifting_modulation);
    
    
    m.def("decimalMap", bind_decimalMap);
    m.def("graycodeword", bind_graycodeword);
    m.def("gray2dec", bind_gray2dec);
    
    m.def("phaseGraycodingUnwrap", bind_phaseGraycodingUnwrap);
}
