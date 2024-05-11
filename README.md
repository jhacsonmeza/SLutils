# Structured Light Utils (SLutils)

This is a collection of utilities implemented in C++ and CUDA of different structured light algorithms for camera-projector systems, with a strong focus on phase unwrapping. You can easily integrate SLutils into your projects for 3D reconstruction or calibration purposes. Many of the structured light techniques implemented in SLutils are described in the book [High-Speed 3D Imaging with Digital Fringe Projection Techniques](https://www.routledge.com/High-Speed-3D-Imaging-with-Digital-Fringe-Projection-Techniques/Zhang/p/book/9780367869724). Current methods supported in this library are:

For phase measurement (wrapped phase estimation):
* N-step phase-shifting algorithm.
* Three-step phase-shifting algorithm.

For phase unwrapping:
* Center line method (using spatial phase unwrapping).
* Phase-shifting + graycoding method.


## ✅ Requirements
* Compiler with C++17 support.
* CMake >= 3.24.
* OpenCV.
* CUDA (optional).
* Ninja (optional but recommended).
* Python >= 3.8 (optional).

This project requires a recent CMake version because it uses the automatic CUDA architecture detection introduced in version 3.24.

You can install OpenCV with:
```bash
sudo apt install libopencv-dev
```
However, consider that if you want to build the CUDA version of SLutils, you will need to build OpenCV from source with the CUDA modules because `libopencv-dev` does not have CUDA support.


## 🛠️ Build from source
To build SLutils from source, first clone this repo:
```bash
$ git clone --recursive https://github.com/jhacsonmeza/SLutils.git
$ cd SLutils
```
Please do not forget the `--recursive` flag, specially if you want to build the Python bindings, because this clones the submodule dependencies.

Now, create a build folder and configure the building process with CMake:
```bash
$ mkdir build && cd build
$ cmake ..
# If you have Ninja installed use: cmake -GNinja ..
```
This will automatically detect if you have a CUDA compiler available to build the CUDA version of SLutils. If you do not have CUDA, then the CPU version of SLutils will be compiled automatically.

After the CMake configuration, you are ready to finally build the library by running:
```bash
$ cmake --build .
```

**Note**: If you have CUDA but you want to build the CPU version of SLutils, you can do this with the `SLU_WITH_CUDA` option. You just need to replace the `cmake ..` command with
```bash
$ cmake -DSLU_WITH_CUDA=OFF ..
```



## ⚙️ CMake options
This is a full list of all the CMake options available and their default values.

| **Option**             | **Description**       | **Default** |
|------------------------|-----------------------|-------------|
| `SLU_WITH_CUDA`        | Build CUDA version    | `ON`        |
| `SLU_BUILD_SAMPLES`    | Build code samples    | `OFF`       |
| `SLU_PYTHON_BINDINGS`  | Build Python bindings | `OFF`       |


## 💻 Code samples
We provide some code samples to try out SLutils. By default these samples are not compiled. To build them together with the library you need to use the `SLU_BUILD_SAMPLES` option:

```bash
$ cmake -DSLU_BUILD_SAMPLES=ON ..
$ cmake --build .
```

To run the sample codes, you first need to download the test dataset here. For the following instructions we will assume you placed the dataset inside the `SLutils/` directory and that you are in the `SLutils/build/` directory.

* In `samples/ps.cpp` we provide a N-step Phase-Shifting (PS) example using the `NStepPhaseShifting_modulation` function, which in addition to the wrapped phase map, provides the data modulation map. Run it with:
    ```bash
    SLutils/build$ ./samples/ps ../data/nPS
    ```
    where `../data/nPS` is the path to the images.

* With `samples/ps+gc.cpp` you can use the phase-shifting + graycoding method, where fringe patterns are used to estimate a wrapped phase map and graycode patterns are used to estimate the fringe order for unwrapping. This method is implemented with the `phaseGraycodingUnwrap` function:
    ```bash
    SLutils/build$ ./samples/ps_gc ../data/PS+GC
    ```
    where `../data/PS+GC` is the path to the images.


## 🐍 Python bindings
SLutils provides Python bindings for both the CPU and CUDA versions. This project uses [nanobind](https://github.com/wjakob/nanobind) to generate the Python bindings. For the bindings it is very important to clone this repo using the `--recursive` flag. In case you forgot, you can just run `git submodule update --init --recursive` to recursively clone all the submodules.

For the CPU bindings, SLutils is designed to work with NumPy in the same way OpenCV-Python uses it for input/output of arrays. For the CUDA version, `torch.Tensor` from [PyTorch](https://github.com/pytorch/pytorch) is used for array manipulation because this is a highly effective and easy-to-install Python tensor container for GPUs.

To enable the build of the Python bindings use the `SLU_PYTHON_BINDINGS` option:

```bash
$ cmake -DSLU_PYTHON_BINDINGS=ON ..
$ cmake --build .
```

This, in most cases, will automatically use your system's Python (the one from `/usr/bin/python`).

If you want to build the bindigs with Python from an anaconda enviroment `<env>`, you first need to activate that enviroment and then provide the path to your anaconda Python executable with the CMake variable `Python_ROOT`. Here an example:

```bash
$ conda activate <env>
$ cmake -DSLU_PYTHON_BINDINGS=ON -DPython_ROOT=/home/<user>/anaconda3/envs/<env>/bin/python ..
$ cmake --build .
```

This is the common path aftert installing anaconda in the home directory. If you are not sure about the path to the Python executable binary, you can verify it with:

```bash
$ conda activate <env>
$ python -c "import sys; print(sys.executable)"
```

This will print the path of the current Python interpreter executable that you need to pass with `Python_ROOT`.
