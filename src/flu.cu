#include <GL/glew.h>    // GLint
#include <GL/glut.h>


// General imports
#include <iostream>
// #include <fstream>
// #include <vector>
// #include <chrono>
// #include <stdlib.h>

// CUDA FFT Libraries
#include <cufft.h>

// CUDA imports
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>    // checkCudaErrors
#include <helper_functions.h>


#include "defines.h"
#include "kernel.cuh"

// OpenGL check functions
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
//


using namespace std;

bool g_bExitESC = false;

//simulation tracking variables
static int click_on = 0;
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

// data for calculations
static float2 *particles = NULL;
static int xp = 0, yp = 0;
static float2 *vhost = NULL, *vdev = NULL;
static float2 *vx = NULL, *vy = NULL;
static int window_h = max(512, DIM), window_w = max(512, DIM);

// GL necessities
GLuint vbo = 0;
struct cudaGraphicsResource *cuda_vbo_resource;
static cudaArray *array = NULL;
// cudaTextureObject_t texObj;
size_t pitch = 0;


char *ref_file = NULL;


// Texture
// texture<float2, 2> texObj;
// static cudaArray *textArray = NULL;

// CUFFT
cufftHandle planr2c;
cufftHandle planc2r;



void fluid_simulation_step() {
    // simple four steps from Stable Fluids paper
    dim3 grid(TW, TH);
    dim3 block(BLOCKDX, BLOCKDY);

    updateTexture(vdev);
    advect_velocity<<<grid, block>>>(vdev, (float *)vx, (float *)vy, DIM, ROWPAD, DT, TW/TH);
    // getLastCudaError("advect_velocity failed");

    // checkCudaErrors(cufftExecR2C(planr2c, (cufftReal *)vx, (cufftComplex *)vx));
    // checkCudaErrors(cufftExecR2C(planr2c, (cufftReal *)vy, (cufftComplex *)vy));

    diffuse_projection<<<grid, block>>>(vx, vy, DIM, DT, VISC, TW/TH);

    // checkCudaErrors(cufftExecC2R(planc2r, (cufftComplex *)vx, (cufftReal *)vx));
    // checkCudaErrors(cufftExecC2R(planc2r, (cufftComplex *)vy, (cufftReal *)vy));

    update_velocity_cpu(vdev, (float *)vx, (float *)vy, DIM, ROWPAD, DT, TW/TH);

    float2 *p;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&p, &num_bytes, cuda_vbo_resource);

    advect_particles_cpu(p, vdev, DIM, DT, TW/TH);

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}


void init_particles(float2 *p, int dx, int dy) {
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
// void keyboard(unsigned char key, int x, int y) {
//     switch (key) {
//         case 'x':
//             g_bExitESC = true;
//             return;
//
//         case 'r':
//             // reinit velocity/position states
//             memset(vhost, 0, sizeof(float2) * DIMSQ);
//             cudaMemcpy(vdev, vhost, sizeof(float2) * DIMSQ, cudaMemcpyHostToDevice);
//             init_particles(particles, DIM, DIM);
//
//             cudaGraphicsUnregisterResource(cuda_vbo_resource);
//             getLastCudaError("cudaGraphicsUnregisterBuffer failed!");
//
//             glBindBuffer(GL_ARRAY_BUFFER, vbo);
//             glBufferData(GL_ARRAY_BUFFER, sizeof(float2)*DIMSQ, particles, GL_DYNAMIC_DRAW_ARB);
//             glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//             cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
//
//             getLastCudaError("cudaGraphicsGLRegisterBuffer failed");
//             break;
//
//         default:
//             break;
//     }
// }


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
    if (!ref_file)
    {
        sdkStartTimer(&timer);
        fluid_simulation_step();
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glColor4f(0.f, 1.f, 0.f, 0.5f);
    glPointSize(1);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    cout << vbo << endl;
    // glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // glVertexPointer(2, GL_FLOAT, 0, NULL);
    // glDrawArrays(GL_POINTS, 0, DIMSQ);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    // glDisableClientState(GL_VERTEX_ARRAY);
    // glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    // glDisable(GL_TEXTURE_2D);

    if (ref_file)
    {
        return;
    }

    // Finish timing before swap buffers to avoid refresh sync
    sdkStopTimer(&timer);
    glutSwapBuffers();

    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", DIM, DIM, ifps);
        glutSetWindowTitle(fps);
        fpsCount = 0;
        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }

    glutPostRedisplay();
}


int initGL(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_w, window_h);
    glutCreateWindow("Stable Fluid Simulation");
    glutDisplayFunc(display);
    glutMainLoop();
    // glutKeyboardFunc(keyboard);
    // glutMouseFunc(click);
    // glutMotionFunc(motion);
    // glutReshapeFunc(reshape);

    return true;
}


/** TESTING HELLO WORLD **/
void displayForGlut(void) {
    //clears the pixels
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_QUADS);
    glVertex3f(0.10, 0.10, 0.0);
    glVertex3f(0.9, 0.10, 0.0);
    glVertex3f(0.9, 0.9, 0.0);
    glVertex3f(0.10, 0.9, 0.0);
    glEnd();
    glFlush();
}

int initGlutDisplay(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(300, 300);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Example 1.0: Hello World!");
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
    glutDisplayFunc(displayForGlut);
    glutMainLoop();
    return 0;
}





int main(int argc, char **argv) {

    // setup OpenGL
    int dev_id;
    GLint bsize;
    cudaDeviceProp deviceProps;

    if (initGL(&argc, argv) == false) exit(1);

    /** TESTING HELLO WORLD **/
    // if (initGlutDisplay(argc, argv) == false) exit(1);

    dev_id = findCudaDevice(argc, (const char **) argv); // attempt to use specified CUDA device

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, dev_id));
    printf("Using CUDA device [%s] (%d processors)\n",
            deviceProps.name,
            deviceProps.multiProcessorCount);

    // ref_file = "data/ref_fluidsGL.ppm";

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // setup host data for simulation
    vhost = (float2 *) malloc(sizeof(float2) * DIMSQ);
    memset(vhost, 0, sizeof(float2)*DIMSQ);

    // setup device data for fluid simulation
    particles = (float2 *) malloc(sizeof(float2)*DIMSQ);
    memset(particles, 0, sizeof(float2)*DIMSQ);
    init_particles(particles, DIM, DIM);

    cudaMallocPitch((void **)&vdev, &pitch, sizeof(float2)*DIM, DIM);
    getLastCudaError("cudaMallocPitch failed");
    cudaMemcpy(vdev, vhost, sizeof(float2)*DIMSQ, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&vx, sizeof(float2) * PADSZ);
    cudaMalloc((void **)&vy, sizeof(float2) * PADSZ);


    setupTexture(DIM, DIM);
    bindTexture();

    // Create CUFFT transform plan configuration
    checkCudaErrors(cufftPlan2d(&planr2c, DIM, DIM, CUFFT_R2C));
    checkCudaErrors(cufftPlan2d(&planc2r, DIM, DIM, CUFFT_C2R));
    // TODO: update kernels to use the new unpadded memory layout for perf
    // rather than the old FFTW-compatible layout
    // cufftSetCompatibilityMode(planr2c, CUFFT_COMPATIBILITY_FFTW_PADDING);
    // cufftSetCompatibilityMode(planc2r, CUFFT_COMPATIBILITY_FFTW_PADDING);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2)*DIMSQ, particles, GL_DYNAMIC_DRAW_ARB);

    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize);

    if (bsize != (sizeof(float2) * DIMSQ)) exit(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
    getLastCudaError("cudaGraphicsGLRegisterBuffer failed");

    for (int ct=0; ct<STEPS; ct++) {
        fluid_simulation_step();

        // addForces(vhost, DIM, DIM, spx, spy);
    }

    return 0;
}
