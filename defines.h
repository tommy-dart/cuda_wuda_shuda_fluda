#ifndef DEFINES_H
#define DEFINES_H

#define DIM = 256
#define DIMSQ = (DIM*DIM)

// padding necessities
#define COLPAD (1 + DIM/2)
#define ROWPAD (2*(COLPAD))
#define PADSZ (DIM*COLPAD)

// solver parameters
#define DT 0.1f     // time delta for iterations
#define VISC 0.01f  // viscosity of water at 20C
#define FORCE (6*DIM)
#define FRADIUS 4
#define STEPS 1e5

// thread organization
#define TW 64
#define TH 64

#endif
