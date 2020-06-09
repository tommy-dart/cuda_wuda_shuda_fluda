#include <GL/glew.h>    // GLint
#include <GL/glut.h>

// General imports
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <math.h>
#include <ctime>
#include <chrono>
// #include <stdlib.h>

// CUDA imports
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>    // checkCudaErrors
#include <helper_functions.h>


#include "defines.h"
#include "kernel.cuh"
#include "func_cpu.cuh"


using namespace std;
using namespace chrono;

typedef complex<float> fcomp;

// GPU or CPU implementation
bool gpu_impl = true;

//simulation tracking variables
static int click_on = 0;
static int xp = 0, yp = 0;
static int window_h = DIM, window_w = DIM;
//
// // data for calculations
static float2 *particles = NULL;
static float2 *particles_dev = NULL;
static float2 *vhost = NULL, *vdev = NULL;

static float2 *vx_fft = NULL, *vy_fft = NULL;
static float2 *vx_fft_host = NULL, *vy_fft_host = NULL;

// GL necessities
GLuint vbo = 0;
static float3 *image = NULL;

// CPU version
static float2 *v = NULL;
static vector<fcomp> vx_comp, vy_comp;

// Timing
static int cur_time_iter = 0;
ofstream out;


void fluid_simulation_step_gpu();
void fluid_simulation_step_cpu();


void init_particles(float2 *p,  int dim) {
    for (int i=0; i<dim; i++){
        for (int j=0; j<dim; j++) {
            uint idx = i * dim + j;
            p[idx].x = (i + 0.5f + ((rand() / (float)RAND_MAX) - 0.5f)) / dim;
            p[idx].y = (j + 0.5f + ((rand() / (float)RAND_MAX) - 0.5f)) / dim;
        }
    }
}


void update_image(float2 *p, float3 *img , int dim) {
    float2 pt;

    for (int i = 0; i < dim; i++) {
        for (int j=0; j<dim; j++) {
            uint i_idx = i * dim + j;
            img[i_idx].z = 0.0f;
            img[i_idx].x = 0.5f;
            img[i_idx].y = 0.1f;
        }
    }

    for (int i=0; i<dim; i++){
        for (int j=0; j<dim; j++) {
            uint p_idx = i * dim + j;
            pt = p[p_idx];

            int pti = (int) (pt.x*dim);
            int ptj = (int) (pt.y*dim);

            int i_idx = pti * dim + ptj;

            if (img[i_idx].z < 1.0)
                img[i_idx].z += 0.33f; // make it blu like watar

        }
    }
}


void init_image(float3 *img,  int dim) {
    for (int i=0; i<dim; i++){
        for (int j=0; j<dim; j++) {
            uint idx = i * dim + j;
            img[idx].x = 0.0;
            img[idx].y = 0.0;
            img[idx].z = 0.0;
        }
    }
}


// (for OpenGL)
// if the user clicks, updates the last clicked location and the click_on status
void click(int button, int updown, int x, int y) {
    xp = x;
    yp = y;
    click_on = !click_on;
}


// Handler for window's re-size event
void reshape(GLsizei width, GLsizei height) {  // GLsizei: non-negative integer
   if (height == 0) height = 1;  // prevent divide by 0

   // Set the viewport (display area) to cover entire application window
   glViewport(0, 0, width, height);

   // Select the aspect ratio of the clipping area to match the viewport
   glMatrixMode(GL_PROJECTION);  // Select the Projection matrix
   glLoadIdentity();             // Reset the Projection matrix
   gluPerspective(45.0, (float)width / (float)height, 0.1, 100.0);

   // Reset the Model-View matrix
   glMatrixMode(GL_MODELVIEW);  // Select the Model-View matrix
   glLoadIdentity();            // Reset the Model-View matrix
}


void motion(int x, int y) {
    // convert mouse to domain
    float fx = xp / ((float) window_w);
    float fy = yp / ((float) window_h);
    int nx = (int) (fx * DIM);
    int ny = (int) (fy * DIM);

    if (click_on &&
        nx < DIM - FRADIUS && ny < DIM - FRADIUS &&
        nx > FRADIUS - 1 && ny > FRADIUS - 1) {
            int dx = x - xp;
            int dy = y - yp;

            fx = FORCE * DT * ((float)dx) / ((float) window_w);
            fy = FORCE * DT * ((float)dy) / ((float) window_h);

            int spx = nx - FRADIUS;
            int spy = ny - FRADIUS;

            if (gpu_impl)
                add_forces_host(vdev, spx, spy, fx, fy);
            else
                add_forces_cpu(v, DIM, spx, spy, fx, fy, FRADIUS);

            xp = x;
            yp = y;
    }

    glutPostRedisplay();
}


void display(void) {
    if (gpu_impl)
        fluid_simulation_step_gpu();
    else
        fluid_simulation_step_cpu();

    update_image(particles, image, DIM);

    // Load up particles

    glShadeModel(GL_SMOOTH);               // Enable smooth shading of color
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);  // Set background (clear) color to white

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, DIM, DIM, 0, GL_RGB,
            GL_FLOAT, image);  // Create texture from image data
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glEnable(GL_TEXTURE_2D);  // Enable 2D texture


    // Draw particles

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear screen and depth buffers

    // Draw cube
    glLoadIdentity();   // Reset the view
    glTranslatef(0.0f, 0.0f, -3.4f);

    if (!gpu_impl)
        glRotatef(270.0f, 0.0f, 0.0f, 1.0f);


    glBegin(GL_QUADS);
       // Front Face
       glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
    glEnd();
    glutSwapBuffers(); // Swap front and back buffers (double buffered mode)

    glutPostRedisplay();
}


int initGl(int *argc, char* argv[]) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_w, window_h);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Fluda!");

    glutDisplayFunc(display);
    glutMouseFunc(click);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);    // Register handler for window re-size

    return true;
}


void fluid_simulation_step_cpu() {
    time_point<system_clock> time_start, time_cmp_start;
    duration<double> velocity_time, diffuse_time, particle_time, elapsed_cmp_time;

    time_cmp_start = system_clock::now();

    time_start = system_clock::now();
    advect_velocity_cpu(v, (float *)vx_fft, (float *)vy_fft, DIM, DT);
    velocity_time = system_clock::now() - time_start;


    copy_f2s_to_comps(vx_fft, vy_fft, vx_comp, vy_comp);
    fft_cpu(vx_comp, false);
    fft_cpu(vy_comp, false);
    copy_comps_to_f2s(vx_comp, vy_comp, vx_fft, vy_fft);

    time_start = system_clock::now();
    diffuse_projection_cpu(vx_fft, vy_fft, DIM, DT, VISC);
    diffuse_time = system_clock::now() - time_start;

    copy_f2s_to_comps(vx_fft, vy_fft, vx_comp, vy_comp);

    fft_cpu(vx_comp, true);
    fft_cpu(vy_comp, true);

    copy_comps_to_f2s(vx_comp, vy_comp, vx_fft, vy_fft);

    update_velocity_cpu(v, (float *)vx_fft, (float *)vy_fft, DIM);

    time_start = system_clock::now();
    advect_particles_cpu(particles, v, DIM, DT);
    particle_time = system_clock::now() - time_start;

    // Computation time testing
    elapsed_cmp_time = system_clock::now() - time_cmp_start;

    if (cur_time_iter < TEST_ITER)
    {
        out << elapsed_cmp_time.count() << " "
            << velocity_time.count() << " "
            << diffuse_time.count() << " "
            << particle_time.count() << endl;
    }
    cur_time_iter++;
}


void fluid_simulation_step_gpu() {
    // simple four steps from Stable Fluids paper

    // Computation time testing
    time_point<system_clock> time_start, time_cmp_start;
    duration<double> velocity_time, diffuse_time, particle_time, elapsed_cmp_time;

    time_cmp_start = system_clock::now();

    time_start = system_clock::now();
    advect_velocity_host(vdev, (float *)vx_fft, (float *)vy_fft);
    velocity_time = system_clock::now() - time_start;


    copy_device_f2s_to_comps(vx_fft, vy_fft, vx_fft_host, vy_fft_host, vx_comp, vy_comp);
    fft_cpu(vx_comp, false);
    fft_cpu(vy_comp, false);
    copy_comps_to_f2s_device(vx_comp, vy_comp, vx_fft, vy_fft, vx_fft_host, vy_fft_host);

    time_start = system_clock::now();
    diffuse_projection_host(vx_fft, vy_fft);
    diffuse_time = system_clock::now() - time_start;

    copy_device_f2s_to_comps(vx_fft, vy_fft, vx_fft_host, vy_fft_host, vx_comp, vy_comp);
    fft_cpu(vx_comp, true);
    fft_cpu(vy_comp, true);
    copy_comps_to_f2s_device(vx_comp, vy_comp, vx_fft, vy_fft, vx_fft_host, vy_fft_host);

    update_velocity_host(vdev, (float *)vx_fft, (float *)vy_fft);

    cudaMemcpy(particles_dev, particles, sizeof(float2)*DIMSQ, cudaMemcpyHostToDevice);

    time_start = system_clock::now();
    advect_particles_host(particles_dev, vdev);
    particle_time = system_clock::now() - time_start;

    cudaMemcpy(particles, particles_dev, sizeof(float2)*DIMSQ, cudaMemcpyDeviceToHost);


    // Computation time testing
    elapsed_cmp_time = system_clock::now() - time_cmp_start;

    if (cur_time_iter < TEST_ITER)
    {
        out << elapsed_cmp_time.count() << " "
            << velocity_time.count() << " "
            << diffuse_time.count() << " "
            << particle_time.count() << endl;
    }

    cur_time_iter++;
}


int main(int argc, char **argv) {

    // Initialize particles
    particles = (float2 *) malloc(sizeof(float2)*DIMSQ);
    memset(particles, 0, sizeof(float2)*DIMSQ);
    init_particles(particles, DIM);

    image = (float3 *) malloc(sizeof(float3)*DIMSQ);
    memset(image, 0, sizeof(float3)*DIMSQ);
    init_image(image, DIM);

    // Initialize interactable GUI
    if (initGl(&argc, argv) == false) exit(1);

    // In GPU case
    if (gpu_impl)
    {
        int dev_id;
        cudaDeviceProp deviceProps;

        dev_id = findCudaDevice(argc, (const char **) argv);

        checkCudaErrors(cudaGetDeviceProperties(&deviceProps, dev_id));
        printf("Using CUDA device [%s] (%d processors)\n",
                deviceProps.name,
                deviceProps.multiProcessorCount);

        vhost = (float2 *) malloc(sizeof(float2) * DIMSQ);
        memset(vhost, 0, sizeof(float2)*DIMSQ);

        vx_fft_host = (float2 *) malloc(sizeof(float2) * DIM_FFT_DATA);
        vy_fft_host = (float2 *) malloc(sizeof(float2) * DIM_FFT_DATA);
        memset(vx_fft_host, 0, sizeof(float2)*DIM_FFT_DATA);
        memset(vy_fft_host, 0, sizeof(float2)*DIM_FFT_DATA);

        cudaMalloc((void **)&vdev, sizeof(float2)*DIMSQ);
        cudaMemcpy(vdev, vhost, sizeof(float2)*DIMSQ, cudaMemcpyHostToDevice);

        cudaMalloc((void **)&vx_fft, sizeof(float2) * DIM_FFT_DATA);
        cudaMalloc((void **)&vy_fft, sizeof(float2) * DIM_FFT_DATA);
        cudaMemcpy(vx_fft, vx_fft_host, sizeof(float2)*DIM_FFT_DATA, cudaMemcpyHostToDevice);
        cudaMemcpy(vy_fft, vy_fft_host, sizeof(float2)*DIM_FFT_DATA, cudaMemcpyHostToDevice);

        cudaMalloc((void **)&particles_dev, sizeof(float2) * DIMSQ);
        cudaMemcpy(particles_dev, particles, sizeof(float2)*DIMSQ, cudaMemcpyHostToDevice);

        getLastCudaError("one of the cudaMallocs failed :(.");

        vx_comp = vector<fcomp>(DIM_FFT_DATA);
        vy_comp = vector<fcomp>(DIM_FFT_DATA);
    }

    // In CPU case
    else
    {
        v = (float2 *) malloc(sizeof(float2) * DIMSQ);
        vx_fft = (float2 *) malloc(sizeof(float2) * DIM_FFT_DATA);
        vy_fft = (float2 *) malloc(sizeof(float2) * DIM_FFT_DATA);

        vx_comp = vector<fcomp>(DIM_FFT_DATA);
        vy_comp = vector<fcomp>(DIM_FFT_DATA);
    }

    string file_name;
    if (gpu_impl)
        file_name = "gpu_time.dat";
    else
        file_name = "cpu_time.dat";

	out.open(file_name.c_str());
    if(out.fail()){
		printf("\ncannot open file %s to record results\n",file_name.c_str());
		return 0;
	}


    out << "total_time adv_vel_time diffuse_time adv_part_time" << endl;
    glutMainLoop();

    return 1;
}
