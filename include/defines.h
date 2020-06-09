#ifndef DEFINES_H
#define DEFINES_H

#define DIM 512
#define DIMSQ (DIM*DIM)

// padding necessities
static int DIM_FFT = 1 + DIM / 2;  // number of FFT components after FFT
static int DIM_FFT_INV = DIM_FFT * 2; // number of inverse FFT coefficients
static int DIM_FFT_DATA = DIM * DIM_FFT;  // total size for FFT data vectors

// solver parameters
#define DT 0.09f     // time delta for iterations
#define VISC 0.025f  // viscosity of water at 20C
#define FORCE (10.0f*DIM)
#define FRADIUS 4
#define STEPS 1e5

// thread organization
#define TW 64
#define TH 64
#define BLOCKDX 64
#define BLOCKDY 4

#define TEST_ITER 100

#endif
