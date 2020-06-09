# Fluid Simulation

GPU Programming and High Performance Computing

COSC 189

GPU League Final Project

Authors: Tommy White and Monika Roznere

## Dependencies

CUDA, OpenGL, GLEW, GLUT

Please check if library are included and linked correctly according to user's system in `CMakeLists.txt`.

## Getting Started

This package has been tested on Ubuntu 16.04 with GPU GeForce GTX 1060 with Max-Q Design.

To run GPU implementation, in `src/flu.cu`, change:
```
bool gpu_impl = true;
```

To run CPU implementation, in `src/flu.cu `, change:
```
bool gpu_impl = false;
```

## Install

To clone:

```
git clone https://github.com/tommy-dart/cuda_wuda_shuda_fluda.git
```

## Build

To build:

```
chmod +x build.sh
./build.sh
```

## Run

```
./cuda_wuda_shuda_fluda
```
