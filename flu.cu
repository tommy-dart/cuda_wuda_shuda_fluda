// General imports
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

// CUDA imports
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>

// OpenGL imports
#include <GL/glew.h>

// Mac has a special glut.h
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// OpenGL check functions
#ifdef WIN32
bool hasOpenGL() return true;
#else
# if defined(__APPLE__) || defined(MACOSX)
bool hasOpenGL() return true;
#else
#include <X11/Xlib.h>
bool hasOpenGL() {
    Display *Xdisplay;

    if (Xdisplay = XOpenDisplay(NULL)){
        XCloseDisplay(Xdisplay);
        return true;
    }
    return false;
}

using namespace std;

namespace name
{
    string title="fluid_dynamics";
    string team="cuda_wuda_shuda";
    string author_1="tommy_white";
    string author_2="mon_rozbeer";
}



void fluid_simulation_step() {
    // simple four steps from Stable Fluids paper
    advect_velocity();
    diffuse_projection();
    update_velocity();
    advect_particles();
}

void init_particles(cData *p, int dx, int dy) {
    for (int i=0; i<dy; i++){
        for (int j=0; j<dx; j++) {
            idx = i*dx + j;
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
            memset(vhost, 0, sizeof(cData) * DS);
            cudaMemcpy(vdev, vhost, sizeof(cData) * DS, cudaMemcpyHostToDevice);
            init_particles(particles, DIM, DIM);

            cudaGraphicsUnregisterResource(cuda_vbo_resource);
            getLastCudaError("cudaGraphicsUnregisterBuffer failed!");

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(cData)*DS, particles, GL_DYNAMIC_DRAW_ARB);
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
    float fx = ;
    float fy = ;
    int nx = (int) (fx * DIM);
    int ny = (int) (fy * DIM);

    if (click_on &&
        nx < DIM - FRADIUS && ny < DIM - FRADIUS &&
        nx > FRADIUS - 1 && ny > FRADIUS - 1) {
            int dx = x - xp;
            int dy = y - yp;

            fx = FORCE * DT * dx / (float) window_w;
            fy = FORCE * DT * dy / (float) window_h;

            int px = nx-FRADIUS;
            int py = ny-FRADIUS;

            add_forces(vdev, DIM, DIM, px, py, fx, fy, FRADIUS);

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


int initGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("Stable Fluid Simulation");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(click);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);

    return true;
}


int main() {

    // setup output file
    string fname = "results.out";
    out.open(fname.c_str());

    if (out.fail()) {
        printf("\n\nUnable to open output file (%s)\n", fname.c_str());
        exit(1);
    }


    // setup OpenGL
    int dev_id;
    cudaDeviceProp deviceProps;

    if (initGL(&argc, argv) == false) exit(1);

    dev_id = findCudaGLDevice(arcs, (const char **) argv); // attempt to use specified CUDA device

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, dev_id));
    printf("Using CUDA device [%s] (%d processors)",
            deviceProps.name,
            deviceProps.multiProcessorCount);

    setupTexture(DIM, DIM);
    bindTexture();

    // setup host data for simulation
    vhost = (cData *) malloc(sizeof(cData) * DS);
    memset(vhost, 0, sizeof(cData)*DS)

    // setup device data for fluid simulation
    particles = (cData *) malloc(sizeof(cData)*DIMSQ);
    init_particles(particles, DIM, DIM);

    cudaMallocPitch((void **) &vdev, &pitch, sizeof(cData)*DIM, DIM);
    cudaMemcpy(vdev, vhost, sizeof(cData)*DS, cudaMemcpyHostToDevice)

    cudaMalloc((void **)&vx, sizeof(cData) * PADSZ);
    cudaMalloc((void **)&vx, sizeof(cData) * PADSZ);


    for (int ct=0; ct<STEPS; ct++) {
        fluid_simulation_step();

        addForces(vhost, DIM, DIM, spx, spy)
    }
    run_da_waves();

    return 0;
}
