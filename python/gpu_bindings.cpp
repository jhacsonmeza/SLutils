#include <phase_unwrap/centerline.hpp>
#include <phase_unwrap/fringe_analysis.hpp>
#include <phase_unwrap/graycoding.hpp>
#include <phase_unwrap/phase_graycoding.hpp>

#include <opencv2/core/cuda.hpp>

#include <vector>
#include <string>
#include <utility> // std::pair

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;


void delete_GpuMat(void* p) noexcept {
    delete reinterpret_cast<cv::cuda::GpuMat*>(p);
}


/* ----------------------- Bindings for fringe_analysis.hpp ----------------------- */
nb::ndarray<nb::pytorch, double> bind_NStepPhaseShifting(const std::vector<std::string>& imgs, int N) {
    // Run core function
    cv::cuda::GpuMat phi;
    sl::NStepPhaseShifting(imgs, phi, N);
    
    // Get output size
    const size_t h = phi.rows, w = phi.cols;
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::cuda::GpuMat(phi), delete_GpuMat);
    
    return {phi.data, {h, w}, owner, {static_cast<int64_t>(phi.step1()), 1},
        nb::dtype<double>(), nb::device::cuda::value};
}

auto bind_NStepPhaseShifting_modulation(const std::vector<std::string>& imgs, int N)
  -> std::pair<nb::ndarray<nb::pytorch, double>, nb::ndarray<nb::pytorch, double>> {
    // Run core function
    cv::cuda::GpuMat phi, mod;
    sl::NStepPhaseShifting_modulation(imgs, phi, mod, N);
    
    // Get output size
    const size_t h = phi.rows, w = phi.cols;
    // Get strides
    int64_t sp = phi.step1(), sm = mod.step1();
    
    // Create capsules for output numpy arrays
    nb::capsule owner_phi(new cv::cuda::GpuMat(phi), delete_GpuMat);
    nb::capsule owner_mod(new cv::cuda::GpuMat(mod), delete_GpuMat);
    
    return {{phi.data, {h, w}, owner_phi, {sp, 1}, nb::dtype<double>(), nb::device::cuda::value},
        {mod.data, {h, w}, owner_mod, {sm, 1}, nb::dtype<double>(), nb::device::cuda::value}};
}

nb::ndarray<nb::pytorch, double> bind_ThreeStepPhaseShifting(const std::vector<std::string>& imgs) {
    // Run core function
    cv::cuda::GpuMat phi;
    sl::ThreeStepPhaseShifting(imgs, phi);
    
    // Get output size
    const size_t h = phi.rows, w = phi.cols;
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::cuda::GpuMat(phi), delete_GpuMat);
    
    return {phi.data, {h, w}, owner, {static_cast<int64_t>(phi.step1()), 1},
        nb::dtype<double>(), nb::device::cuda::value};
}

auto bind_ThreeStepPhaseShifting_modulation(const std::vector<std::string>& imgs)
  -> std::pair<nb::ndarray<nb::pytorch, double>, nb::ndarray<nb::pytorch, double>> {
    // Run core function
    cv::cuda::GpuMat phi, mod;
    sl::ThreeStepPhaseShifting_modulation(imgs, phi, mod);
    
    // Get output size
    const size_t h = phi.rows, w = phi.cols;
    // Get strides
    int64_t sp = phi.step1(), sm = mod.step1();
    
    // Create capsules for output numpy arrays
    nb::capsule owner_phi(new cv::cuda::GpuMat(phi), delete_GpuMat);
    nb::capsule owner_mod(new cv::cuda::GpuMat(mod), delete_GpuMat);
    
    return {{phi.data, {h, w}, owner_phi, {sp, 1}, nb::dtype<double>(), nb::device::cuda::value},
        {mod.data, {h, w}, owner_mod, {sm, 1}, nb::dtype<double>(), nb::device::cuda::value}};
}


/* ----------------------- Bindings for graycoding.hpp ----------------------- */
nb::ndarray<nb::pytorch, int> bind_decimalMap(const std::vector<std::string>& imlist) {
    // Run core function
    cv::cuda::GpuMat dec;
    sl::decimalMap(imlist, dec); // returns int 2D array
    
    // Get output size
    const size_t h = dec.rows, w = dec.cols;
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::cuda::GpuMat(dec), delete_GpuMat);
    
    return {dec.data, {h, w}, owner, {static_cast<int64_t>(dec.step1()), 1},
        nb::dtype<int>(), nb::device::cuda::value};
}

nb::ndarray<nb::pytorch, int> bind_gray2dec(nb::ndarray<uchar, nb::ndim<3>, nb::device::cuda> _code_word) {
    // Get size of the input array
    const size_t n = _code_word.shape(0), h = _code_word.shape(1), w = _code_word.shape(2);
    
    // Create view of the input 3D array in the form of std::vector<cv::cuda::GpuMat>
    std::vector<cv::cuda::GpuMat> code_word(n);
    for (int i = 0; i < n; i++) {
        uchar* p = _code_word.data() + i*w*h; // pointer to the i-th slide
        code_word[i] = {static_cast<int>(h), static_cast<int>(w), CV_8U, p/*, _code_word.stride(1)*/};
    }
    
    // Run core function
    cv::cuda::GpuMat dec;
    sl::gray2dec(code_word, dec); // returns int 2D array
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::cuda::GpuMat(dec), delete_GpuMat);
    
    return {dec.data, {h, w}, owner, {static_cast<int64_t>(dec.step1()), 1},
        nb::dtype<int>(), nb::device::cuda::value};
}


/* ----------------------- Bindings for phase_graycoding.hpp ----------------------- */
nb::ndarray<nb::pytorch, double> bind_phaseGraycodingUnwrap(const std::vector<std::string>& imlist_ps,
                                                            const std::vector<std::string>& imlist_gc,
                                                            int p, int N) {
    
    // Run core function
	cv::cuda::GpuMat Phi;
	sl::phaseGraycodingUnwrap(imlist_ps, imlist_gc, Phi, p, N);
    
    // Get output size
    const size_t h = Phi.rows, w = Phi.cols;
    
    // Create capsule for the output numpy array
    nb::capsule owner(new cv::cuda::GpuMat(Phi), delete_GpuMat);
    
    return {Phi.data, {h, w}, owner, {static_cast<int64_t>(Phi.step1()), 1},
        nb::dtype<double>(), nb::device::cuda::value};
}



/////////////////////////////////////////////////////////////////////
/* ----------------------- Create bindings ----------------------- */
/////////////////////////////////////////////////////////////////////
NB_MODULE(sl, m) {
    m.def("NStepPhaseShifting", bind_NStepPhaseShifting);
    m.def("NStepPhaseShifting_modulation", bind_NStepPhaseShifting_modulation);
    m.def("ThreeStepPhaseShifting", bind_ThreeStepPhaseShifting);
    m.def("ThreeStepPhaseShifting_modulation", bind_ThreeStepPhaseShifting_modulation);
    
    
    m.def("decimalMap", bind_decimalMap);
    m.def("gray2dec", bind_gray2dec);
    
    m.def("phaseGraycodingUnwrap", bind_phaseGraycodingUnwrap);
}
