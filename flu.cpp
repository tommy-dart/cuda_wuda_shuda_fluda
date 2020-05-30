#include <helper_gl.h>


// Mac has a special glut.h
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// General imports
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <stdlib.h>

// CUDA imports
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "defines.h"

// // OpenGL check functions
// #ifdef WIN32
// bool hasOpenGL() return true;
// #else
// #if defined(__APPLE__) || defined(MACOSX)
// bool hasOpenGL() return true;
// #else
// #include <X11/Xlib.h>
// bool hasOpenGL() {
//     Display *Xdisplay;
//
//     if (Xdisplay = XOpenDisplay(NULL)){
//         XCloseDisplay(Xdisplay);
//         return true;
//     }
//     return false;
// }
// #endif
// #endif

bool g_bExitESC = false;

using namespace std;

namespace name
{
    string title="fluid_dynamics";
    string team="cuda_wuda_shuda";
    string author_1="tommy_white";
    string author_2="mon_rozbeer";
}

//simulation tracking variables
static int click_on = 0;

// data for calculations
static double2 *particles = NULL;
static int xp = 0, yp = 0;
static double2 *vhost = NULL, *vdev = NULL;
static double2 *vx = NULL, *vy = NULL;
static int window_h = max(512, DIM), window_w = max(512, DIM);

// GL necessities
GLuint vbo = 0;
struct cudaGraphicsResource *cuda_vbo_resource;
static cudaArray *array = NULL;
cudaTextureObject_t texObj;
size_t pitch = 0;


void fluid_simulation_step() {
    // simple four steps from Stable Fluids paper
    // advect_velocity();
    // diffuse_projection();
    // update_velocity();
    // advect_particles();
}

void init_particles(double2 *p, int dx, int dy) {
    for (int i=0; i<dy; i++){
        for (int j=0; j<dx; j++) {
            uint idx = i*dx + j;
            p[idx].x = (j + rand())/dx;
            p[idx].y = (i + rand())/dy;
        }
    }
}


// (for OpenGL)
// keyboard handling: x == exit, r == reset
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 'x':
            g_bExitESC = true;
            return;

        case 'r':
            // reinit velocity/position states
            memset(vhost, 0, sizeof(double2) * DIMSQ);
            cudaMemcpy(vdev, vhost, sizeof(double2) * DIMSQ, cudaMemcpyHostToDevice);
            init_particles(particles, DIM, DIM);

            cudaGraphicsUnregisterResource(cuda_vbo_resource);
            getLastCudaError("cudaGraphicsUnregisterBuffer failed!");

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(double2)*DIMSQ, particles, GL_DYNAMIC_DRAW_ARB);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
    }
}


// (for OpenGL)
// if the user clicks, updates the last clicked location and the click_on status
void click(int button, int updown, int x, int y) {
    xp = x;
    yp = y;
    click_on = !click_on;
}


// (for OpenGL)
// handle mouse motion to generate force @mouse if clicked
void motion(int x, int y) {
    // convert mouse to domain
    double fx = xp / ((double) window_w);
    double fy = yp / ((double) window_h);
    int nx = (int) (fx * DIM);
    int ny = (int) (fy * DIM);

    if (click_on &&
        nx < DIM - FRADIUS && ny < DIM - FRADIUS &&
        nx > FRADIUS - 1 && ny > FRADIUS - 1) {
            int dx = x - xp;
            int dy = y - yp;

            fx = FORCE * DT * dx / (double) window_w;
            fy = FORCE * DT * dy / (double) window_h;

            int px = nx-FRADIUS;
            int py = ny-FRADIUS;

            // add_forces(vdev, DIM, DIM, px, py, fx, fy, FRADIUS);

            xp = x;
            yp = y;
    }

    glutPostRedisplay();
}


// (for OpenGL)
// reshape window function
void reshape(int x, int y) {
    window_w = x;
    window_h = y;
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0,1,1,0,0,1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}


// (for OpenGL)
// display window function
void display(void) {
    fluid_simulation_step();

    glClear(GL_COLOR_BUFFER_BIT);
    glColor4f(0.f, 1.f, 0.f, 0.5f);
    glPointSize(1);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, NULL);
    glDrawArrays(GL_POINTS, 0, DIMSQ);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);

    glutPostRedisplay();
}


int initGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_w, window_h);
    glutCreateWindow("Stable Fluid Simulation");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(click);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);

    return true;
}

void setupTexture(int x, int y)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

    cudaMallocArray(&array, &desc, y, x);
    getLastCudaError("cudaMalloc failed");

    cudaResourceDesc            texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));

    texRes.resType            = cudaResourceTypeArray;
    texRes.res.array.array    = array;

    cudaTextureDesc             texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL));
}

int main(int argc, char **argv) {

    // setup output file
    // string fname = "results.out";
    // out.open(fname.c_str());
    //
    // if (out.fail()) {
    //     printf("\n\nUnable to open output file (%s)\n", fname.c_str());
    //     exit(1);
    // }


    // setup OpenGL
    int dev_id;
    cudaDeviceProp deviceProps;

    if (initGL(&argc, argv) == false) exit(1);

    GLint bsize;
    dev_id = findCudaDevice(argc, (const char **) argv); // attempt to use specified CUDA device

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, dev_id));
    printf("Using CUDA device [%s] (%d processors)",
            deviceProps.name,
            deviceProps.multiProcessorCount);

    // setup host data for simulation
    vhost = (double2 *) malloc(sizeof(double2) * DIMSQ);
    memset(vhost, 0, sizeof(double2)*DIMSQ);

    // setup device data for fluid simulation
    particles = (double2 *) malloc(sizeof(double2)*DIMSQ);
    init_particles(particles, DIM, DIM);

    cudaMallocPitch((void **) &vdev, &pitch, sizeof(double2)*DIM, DIM);
    cudaMemcpy(vdev, vhost, sizeof(double2)*DIMSQ, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&vx, sizeof(double2) * PADSZ);
    cudaMalloc((void **)&vy, sizeof(double2) * PADSZ);


    setupTexture(DIM, DIM);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(double2)*DIMSQ, particles, GL_DYNAMIC_DRAW_ARB);

    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

    if (bsize != (sizeof(double2) * DIMSQ)) exit(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
    getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

    for (int ct=0; ct<STEPS; ct++) {
        fluid_simulation_step();

        // addForces(vhost, DIM, DIM, spx, spy);
    }

    return 0;
}
