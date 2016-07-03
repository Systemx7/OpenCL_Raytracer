//
//  main.cpp
//  OpenCL_Raytracer
//
//  Created by Stefan Matthes on 03.07.15.
//  Copyright (c) 2015 Stefan Matthes. All rights reserved.
//

////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#include <glew.h>
#include <glfw3.h>
#include <OpenGL/OpenGL.h>
#include <OpenCL/opencl.h>

#include "opencv2/highgui/highgui.hpp"

////////////////////////////////////////////////////////////////////////////////

#define COMPUTE_KERNEL_FILENAME                 ("./raytracer.cl")
#define COMPUTE_KERNEL_METHOD_NAME              ("raytracer")

GLFWwindow* window                              = NULL;
static int window_width                         = 0;
static int window_height                        = 0;

static int render_buffer_width                  = 512;//1024;
static int render_buffer_height                 = 320;//640;

static GLuint fboId                             = 0;
static GLuint rboId                             = 0;

static float desired_camera_position[4]         = { 0.f, 0.f, 0.f, 0.f  };
static float desired_camera_rotation[4]         = { 0.f, 0.f, 0.f, 0.f  };

static float camera_position[4]                 = { 0.f, 0.f, -1.f, 0.f };
static float camera_rotation[4]                 = { 0.f, 0.f, 0.f, 0.f  };

static float old_cursor_pos_x                   = 0;
static float old_cursor_pos_y                   = 0;

static int button_state                         = 0;
const float inertia                             = 0.2;

char time_stats[1024]                           = "\0";

////////////////////////////////////////////////////////////////////////////////

typedef struct QuadNode
{
    cl_float bbox[2*3*4];
    cl_uint child_nodes[4];
    cl_uint number_of_spheres[4];
}QuadNode;

static QuadNode* qbvh_array;
static cl_mem qbvh_buffer;

static unsigned int sphere_count = 0;
static unsigned int first_light_index = 0;
typedef struct Sphere
{
    cl_float pos_and_r[4];
    cl_float color[4];
    cl_uint texture_index;
    cl_float reflection;
    cl_float refraction;
    cl_float eta;
}Sphere;

static Sphere* sphere_array;
static cl_mem sphere_buffer;
static cl_mem sphere_count_buffer;
static cl_mem first_light_index_buffer;

static cl_mem camera_transform_buffer;
static cl_mem seed_buffer;
static cl_mem render_buffer;

static CGLContextObj opengl_context;
static cl_context compute_context;
static cl_kernel compute_kernel;
static cl_program compute_program;
static cl_command_queue command_queue;
static cl_device_id compute_device_id;

static size_t max_work_group_size;

////////////////////////////////////////////////////////////////////////////////

static int fileToString(const char *file_name, char **result_string, size_t *string_len)
{
    int fd;
    unsigned file_len;
    struct stat file_status;
    int ret;
    
    *string_len = 0;
    fd = open(file_name, O_RDONLY);
    if (fd == -1)
    {
        printf("Error opening file %s\n", file_name);
        return -1;
    }
    ret = fstat(fd, &file_status);
    if (ret)
    {
        printf("Error reading status for file %s\n", file_name);
        return -1;
    }
    file_len = (unsigned)file_status.st_size;
    
    *result_string = (char*)calloc(file_len + 1, sizeof(char));
    ret = (int)read(fd, *result_string, file_len);
    if (!ret)
    {
        printf("Error reading from file %s\n", file_name);
        return -1;
    }
    
    close(fd);
    
    *string_len = file_len;
    return 0;
}

////////////////////////////////////////////////////////////////////////////////

static int raytrace(void)
{
    volatile int err = 0;
    // Generate rendum numbers for every pixel for each frame
    int pixel_count = render_buffer_height*render_buffer_width;
    unsigned int* random_numbers;
    random_numbers = (unsigned int*)malloc(sizeof(cl_uint) * pixel_count);
     
    for( int i=0; i<pixel_count; i++)
        random_numbers[i] = rand() % (1<<16);
     
    err = clEnqueueWriteBuffer(command_queue, seed_buffer, CL_TRUE, 0, sizeof(cl_uint)*pixel_count, random_numbers, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to seed_buffer!\n");
        return err;
    }
    free(random_numbers);
    
    // Smooth update of camera pose
    for(int i=0; i<3; ++i)
    {
        camera_position[i] += (desired_camera_position[i] - camera_position[i]) * inertia;
        camera_rotation[i] += (desired_camera_rotation[i] - camera_rotation[i]) * inertia;
    }
    
    float camera_pose[12] = {static_cast<float>(cos(camera_rotation[1])),
        static_cast<float>(sin(camera_rotation[0])*sin(camera_rotation[1])),
        static_cast<float>(cos(camera_rotation[0])*sin(camera_rotation[1])),
        camera_position[0],
        0.f,
        static_cast<float>(cos(camera_rotation[0])),
        static_cast<float>(-sin(camera_rotation[0])),
        camera_position[1],
        static_cast<float>(-sin(camera_rotation[1])),
        static_cast<float>(sin(camera_rotation[0])*cos(camera_rotation[1])),
        static_cast<float>(cos(camera_rotation[0])*cos(camera_rotation[1])),
        camera_position[2]};
    
    err = clEnqueueWriteBuffer(command_queue, camera_transform_buffer, CL_TRUE, 0, sizeof(cl_float)*12, camera_pose, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to camera_transform!\n");
        return err;
    }
    
    err = clEnqueueAcquireGLObjects(command_queue, 1, &render_buffer, 0, 0, 0);
    if (err != CL_SUCCESS)
    {
        printf("Failed to attach Render Buffer!\n");
        return err;
    }
    
    
    err = CL_SUCCESS;
    err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &qbvh_buffer);
    err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &sphere_buffer);
    err |= clSetKernelArg(compute_kernel, 2, sizeof(cl_mem), &sphere_count_buffer);
    err |= clSetKernelArg(compute_kernel, 3, sizeof(cl_mem), &first_light_index_buffer);
    err |= clSetKernelArg(compute_kernel, 4, sizeof(cl_mem), &camera_transform_buffer);
    err |= clSetKernelArg(compute_kernel, 5, sizeof(cl_mem), &seed_buffer);
    err |= clSetKernelArg(compute_kernel, 6, sizeof(cl_mem), &render_buffer);
    if(err)
        return err;
    
    size_t global[2];
    global[0] = render_buffer_width;
    global[1] = render_buffer_height;
    
    size_t local[2] = {32, 16};
    
    err = clEnqueueNDRangeKernel(command_queue, compute_kernel, 2, NULL, global, local, 0, NULL, NULL);
    if(err)
        return err;
    
    err = clEnqueueReleaseGLObjects(command_queue, 1, &render_buffer, 0, 0, 0);
    if (err != CL_SUCCESS)
        return err;
    
    clFlush(command_queue);
    
    return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////

static int setupComputeDevices()
{
    int err;
    printf("Using active OpenGL context...\n");
    
    CGLContextObj kCGLContext = CGLGetCurrentContext();
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
    
    cl_context_properties properties[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties)kCGLShareGroup, 0
    };
    
    // Create a context from a CGL share group
    compute_context = clCreateContext(properties, 0, 0, clLogMessagesToStdoutAPPLE, 0, 0);
    if(!compute_context)
    {
        printf("Error: Failed to create compute context!\n");
        return EXIT_FAILURE;
    }
    
    unsigned int device_count;
    cl_device_id device_ids[16];
    
    size_t returned_size;
    err = clGetContextInfo(compute_context, CL_CONTEXT_DEVICES, sizeof(device_ids), device_ids, &returned_size);
    if(err)
    {
        printf("Error: Failed to retrieve compute devices for context!\n");
        return EXIT_FAILURE;
    }
    
    device_count = (unsigned)returned_size / sizeof(cl_device_id);
    
    
    bool device_found = false;
    cl_device_type device_type;
    for(int i=0; i<device_count; i++)
    {
        clGetDeviceInfo(device_ids[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
        if(device_type == CL_DEVICE_TYPE_GPU)
        {
            compute_device_id = device_ids[i];
            device_found = true;
            break;
        }
    }
    
    if(!device_found)
    {
        printf("Error: Failed to locate compute device!\n");
        return EXIT_FAILURE;
    }
    
    // Create a command queue
    command_queue = clCreateCommandQueue(compute_context, compute_device_id, 0, &err);
    if (!command_queue)
    {
        printf("Error: Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }
    
    // Report the device vendor and device name
    cl_char vendor_name[1024] = {};
    cl_char device_name[1024] = {};
    size_t max_cu = 0;
    err = clGetDeviceInfo(compute_device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
    err|= clGetDeviceInfo(compute_device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
    err|= clGetDeviceInfo(compute_device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_cu), &max_cu, &returned_size);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve device info!\n");
        return EXIT_FAILURE;
    }
    
    printf("Connecting to %s %s...\n", vendor_name, device_name);
    printf("Device has %ld compute units...\n", max_cu);
    
    return CL_SUCCESS;
}

static int setupComputeMemory()
{
    int err;
    printf("Allocating buffers on compute device...\n");
    
    qbvh_buffer = clCreateBuffer(compute_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(QuadNode), qbvh_array, &err);
    if (!qbvh_buffer || err != CL_SUCCESS)
    {
        printf("Failed to create qbvh_buffer! %d\n", err);
        return EXIT_FAILURE;
    }
    
    sphere_buffer = clCreateBuffer(compute_context, CL_MEM_READ_ONLY, sizeof(Sphere)*sphere_count, NULL, &err);
    if (!sphere_buffer || err != CL_SUCCESS)
    {
        printf("Failed to create sphere_buffer! %d\n", err);
        return EXIT_FAILURE;
    }
    
    sphere_count_buffer = clCreateBuffer(compute_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &sphere_count, &err);
    if (!sphere_count_buffer || err != CL_SUCCESS)
    {
        printf("Failed to create sphere_count_buffer! %d\n", err);
        return EXIT_FAILURE;
    }
    
    first_light_index_buffer = clCreateBuffer(compute_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &first_light_index, &err);
    if (!first_light_index_buffer || err != CL_SUCCESS)
    {
        printf("Failed to create first_light_index_buffer! %d\n", err);
        return EXIT_FAILURE;
    }
    
    camera_transform_buffer = clCreateBuffer(compute_context, CL_MEM_READ_ONLY, sizeof(cl_float)*12, NULL, &err);
    if (!camera_transform_buffer || err != CL_SUCCESS)
    {
        printf("Failed to create camera_transform_buffer! %d\n", err);
        return EXIT_FAILURE;
    }
    
    seed_buffer = clCreateBuffer(compute_context, CL_MEM_READ_ONLY, sizeof(cl_uint)*render_buffer_width*render_buffer_height, NULL, &err);
    if (!seed_buffer || err != CL_SUCCESS)
    {
        printf("Failed to create seed_buffer! %d\n", err);
        return EXIT_FAILURE;
    }
    
    render_buffer = clCreateFromGLRenderbuffer(compute_context, CL_MEM_READ_WRITE, rboId, &err);
    if (!render_buffer || err != CL_SUCCESS)
    {
        printf("Failed to create render_buffer! %d\n", err);
        return EXIT_FAILURE;
    }
    
    err = clEnqueueWriteBuffer(command_queue, sphere_buffer, CL_TRUE, 0, sizeof(Sphere)*sphere_count, sphere_array, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to sphere_buffer!\n");
        return EXIT_FAILURE;
    }
    
    
    int pixel_count = render_buffer_height*render_buffer_width;
    unsigned int* random_numbers;
    random_numbers = (unsigned int*)malloc(sizeof(cl_uint) * pixel_count);
    
    for( int i=0; i<pixel_count; i++)
    random_numbers[i] = rand() % (1<<16);
    
    err = clEnqueueWriteBuffer(command_queue, seed_buffer, CL_TRUE, 0, sizeof(cl_uint)*pixel_count, random_numbers, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to seed_transform!\n");
        return EXIT_FAILURE;
    }
    free(random_numbers);
    
    return CL_SUCCESS;
}

static int setupComputeKernels(void)
{
    int err = 0;
    char *source = 0;
    size_t length = 0;
    
    printf("Loading kernel source from file '%s'...\n", COMPUTE_KERNEL_FILENAME);
    err = fileToString(COMPUTE_KERNEL_FILENAME, &source, &length);
    if (err)
        return err;
    
    // Create the compute program from the source buffer
    compute_program = clCreateProgramWithSource(compute_context, 1, (const char **) & source, NULL, &err);
    if (!compute_program || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute program! %d\n", err);
        return EXIT_FAILURE;
    }
    
    // Build the program executable
    printf("Building compute program...\n");
    err = clBuildProgram(compute_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(compute_program, compute_device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }
    
    // Create the compute kernel from within the program
    printf("Creating kernel '%s'...\n", COMPUTE_KERNEL_METHOD_NAME);
    compute_kernel = clCreateKernel(compute_program, COMPUTE_KERNEL_METHOD_NAME, &err);
    if (!compute_kernel|| err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }
    
    // Get the maximum work group size for executing the kernel on the device
    size_t max_wg_size = 1;
    size_t wg_size     = 0;
    err = clGetKernelWorkGroupInfo(compute_kernel, compute_device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg_size), &max_wg_size, NULL);
    err|= clGetKernelWorkGroupInfo(compute_kernel, compute_device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(wg_size), &wg_size, NULL);
    
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return EXIT_FAILURE;
    }
    
    max_work_group_size = max_wg_size;
    printf("Maximum workgroup size '%ld'\n", max_work_group_size);
    printf("Preferred workgroup size multiple: %ld\n", wg_size);
    
    return CL_SUCCESS;
}

static int initOpenCL()
{
    printf("Setting up OpenCL...\n");
    
    int err;
    err = setupComputeDevices();
    if (err != CL_SUCCESS)
    return err;
    
    err = setupComputeMemory();
    if (err != CL_SUCCESS)
    return err;
    
    err = setupComputeKernels();
    if (err != CL_SUCCESS)
    return err;
    
    return CL_SUCCESS;
}


static void shutdownOpenCL(void)
{
    clFinish(command_queue);
    
    clReleaseMemObject(render_buffer);
    clReleaseMemObject(qbvh_buffer);
    clReleaseMemObject(sphere_buffer);
    free(sphere_array);
    clReleaseMemObject(seed_buffer);
    clReleaseMemObject(camera_transform_buffer);
    
    clReleaseKernel(compute_kernel);
    clReleaseProgram(compute_program);
    clReleaseContext(compute_context);
    
    free(qbvh_array);
    
}

////////////////////////////////////////////////////////////////////////////////


int initOpenGL()
{
    printf("Setting up OpenGL...\n");
    
    opengl_context = CGLGetCurrentContext();
    
    glGenFramebuffers(1, &fboId);
    glBindFramebuffer(GL_FRAMEBUFFER, fboId);
    
    glGenRenderbuffers(1, &rboId);
    glBindRenderbuffer(GL_RENDERBUFFER, rboId);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, render_buffer_width, render_buffer_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rboId);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    
    return GL_NO_ERROR;
}


////////////////////////////////////////////////////////////////////////////////

static void showComputationTime(double elapsed_time, int frame_count)
{
    double comp_time_in_ms = elapsed_time * 1000.0 /double(frame_count);
    double fps = double(frame_count) / elapsed_time;
    
    sprintf(time_stats, "Computation time: %3.2lf ms  Display: %3.2lf fps\n", comp_time_in_ms, fps);
    
    glfwSetWindowTitle(window, time_stats);
}

void display(void)
{
    int err = raytrace();
    if (err != 0)
    {
        printf("error %d during raytracing!\n", err);
        shutdownOpenCL();
        exit(1);
    }
    
    glBlitFramebuffer(0, 0, render_buffer_width, render_buffer_height, 0, 0, window_width, window_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void mouseMoveCallback(GLFWwindow* window, double x, double y)
{
    float dx, dy;
    dx = (float)x - old_cursor_pos_x;
    dy = (float)y - old_cursor_pos_y;
    
    switch (button_state) {//seems to work for touchpad...
        case 1: // rotate
            desired_camera_rotation[0] += dy*0.01f;
            desired_camera_rotation[1] += dx*0.01f;
            break;
        case 2: // translate
            desired_camera_position[0] += cos(camera_rotation[1])*dx*0.01f;
            desired_camera_position[1] -= dy*0.01f;
            desired_camera_position[2] -= sin(camera_rotation[1])*dx*0.01f;
            break;
        case 3: // zoom
            //TODO
            break;
        default:
            break;
    }
    old_cursor_pos_x = x;
    old_cursor_pos_y = y;
}

void mousePressCallback(GLFWwindow* window, int button, int state, int mods)
{
    if (state == GLFW_PRESS)
        button_state |= 1 << button;
    else if (state == GLFW_RELEASE)
        button_state = 0;
    
    if (mods & GLFW_MOD_SHIFT)
        button_state = 2;
    else if (mods & GLFW_MOD_CONTROL)
        button_state = 3;
}

void windowReshapeCallback(GLFWwindow* window, int w, int h)
{
    window_width = w;
    window_height = h;
}


void loadScene()
{
    sphere_count = 17;
    sphere_array = (Sphere*)malloc(sizeof(Sphere) * sphere_count);
    float colors[16][4] = {1.f, 0.f, 0.f, 0.f,
                           0.f, 1.f, 0.f, 0.f,
                           0.f, 0.f, 1.f, 0.f,
                           2.f, 0.f, 1.5f, 0.f,
                           2.6f, 2.0f, 0.f, 0.f,
                           0.f, 1.f, 1.9f, 0.f,
                           1.f, 0.3f, 1.f, 0.f,
                           0.f, 1.f, 0.1f, 0.f,
                           1.f, 0.f, 0.f, 0.f,
                           0.f, 1.f, 0.f, 0.f,
                           0.f, 0.f, 1.f, 0.f,
                           2.f, 0.f, 1.5f, 0.f,
                           2.6f, 2.0f, 0.f, 0.f,
                           0.f, 1.f, 1.9f, 0.f,
                           1.f, 0.3f, 1.f, 0.f,
                           0.f, 1.f, 0.1f, 0.f};
    int index = 0;
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            sphere_array[index].pos_and_r[0] = float(2*i-3);
            sphere_array[index].pos_and_r[1] = -2.f;
            sphere_array[index].pos_and_r[2] = float(2*j+2);
            sphere_array[index].pos_and_r[3] = 1.f;
            
            sphere_array[index].color[0] = colors[i*4+j][0];
            sphere_array[index].color[1] = colors[i*4+j][1];
            sphere_array[index].color[2] = colors[i*4+j][2];
            sphere_array[index].color[3] = colors[i*4+j][3];
            
            sphere_array[index].texture_index = 0;
            sphere_array[index].reflection    = 0.3f;
            sphere_array[index].refraction    = 0.f;
            sphere_array[index].eta           = 0.f;
            
            ++index;
        }
    }
    
    first_light_index = 16;
    sphere_array[first_light_index].pos_and_r[0] = -5.f;
    sphere_array[first_light_index].pos_and_r[1] = 2.f;
    sphere_array[first_light_index].pos_and_r[2] = 5.f;//1.f;
    sphere_array[first_light_index].pos_and_r[3] = 0.1f;
    sphere_array[first_light_index].color[0] = 3.f;
    sphere_array[first_light_index].color[1] = 3.f;
    sphere_array[first_light_index].color[2] = 3.f;
    sphere_array[first_light_index].color[3] = 0.35f;
    
    qbvh_array = (QuadNode*)malloc(sizeof(QuadNode));
    qbvh_array[0].bbox[0] = -4.f;
    qbvh_array[0].bbox[1] = -2.f;
    qbvh_array[0].bbox[2] = 0.f;
    qbvh_array[0].bbox[3] = 2.f;
    qbvh_array[0].bbox[4] = -2.f;
    qbvh_array[0].bbox[5] = 0.f;
    qbvh_array[0].bbox[6] = 2.f;
    qbvh_array[0].bbox[7] = 4.f;
    
    qbvh_array[0].bbox[8] = -3.f;
    qbvh_array[0].bbox[9] = -3.f;
    qbvh_array[0].bbox[10] = -3.f;
    qbvh_array[0].bbox[11] = -3.f;
    qbvh_array[0].bbox[12] = -1.f;
    qbvh_array[0].bbox[13] = -1.f;
    qbvh_array[0].bbox[14] = -1.f;
    qbvh_array[0].bbox[15] = -1.f;
    
    qbvh_array[0].bbox[16] = 1.f;
    qbvh_array[0].bbox[17] = 1.f;
    qbvh_array[0].bbox[18] = 1.f;
    qbvh_array[0].bbox[19] = 1.f;
    qbvh_array[0].bbox[20] = 11.f;
    qbvh_array[0].bbox[21] = 11.f;
    qbvh_array[0].bbox[22] = 11.f;
    qbvh_array[0].bbox[23] = 11.f;
    
    qbvh_array[0].child_nodes[0] = 0;
    qbvh_array[0].child_nodes[1] = 4;
    qbvh_array[0].child_nodes[2] = 8;
    qbvh_array[0].child_nodes[3] = 12;
    
    qbvh_array[0].number_of_spheres[0] = 4;
    qbvh_array[0].number_of_spheres[1] = 4;
    qbvh_array[0].number_of_spheres[2] = 4;
    qbvh_array[0].number_of_spheres[3] = 4;
    
    return;
}


int main(int argc, char** argv)
{
    //Record video, if file name is passed as an argument
    cv::Mat frame( render_buffer_height, render_buffer_width, CV_8UC3 );
    cv::VideoWriter vid;
    bool write_video_to_file = false;
    if(argc > 1)
    {
        if(argc == 2)
        {
            char* file_name_with_ext;
            file_name_with_ext = (char*)malloc(strlen(argv[1])+5);
            strcpy(file_name_with_ext, argv[1]);
            strcat(file_name_with_ext, ".avi");
            
            cv::Size frame_size(render_buffer_width, render_buffer_height);
            vid.open(file_name_with_ext, CV_FOURCC('8','B','P','S'), 20, frame_size, true);
            free(file_name_with_ext);
            if( !vid.isOpened() )
            {
                printf("Failed to open video!");
                return -1;
            }
            write_video_to_file = true;
        }
        else
        {
            printf("Error: Too many input arguments...");
            return -1;
        }
    }
    
    // Initialise GLFW
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        return -1;
    }
    
    // Open a window and create its OpenGL context
    const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    window = glfwCreateWindow(mode->width, mode->height, "", NULL, NULL);
    if( window == NULL )
    {
        fprintf( stderr, "Failed to open GLFW window.\n" );
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    
    if( glewInit() != GLEW_OK )
    {
        fprintf( stderr, "Failed to initialize GLEW\n" );
        return -1;
    }
    
    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    
    glfwGetFramebufferSize(window, &window_width, &window_height);
    glfwSetFramebufferSizeCallback(window, windowReshapeCallback);
    glfwSetMouseButtonCallback(window, mousePressCallback);
    glfwSetCursorPosCallback(window, mouseMoveCallback);
    glfwSwapInterval(2);
    
    loadScene(); //Load all the spheres!!!
    
    int err;
    err = initOpenGL();
    if (err != GL_NO_ERROR)
    {
        printf("Failed to initialize OpenGL! Error %d\n", err);
        exit (err);
    }
    
    err = initOpenCL();
    if (err != GL_NO_ERROR)
    {
        printf("Failed to initialize OpenCL! Error %d\n", err);
        exit (err);
    }
    
    // Measure fps...
    double cur_time, old_time;
    cur_time = old_time = glfwGetTime();
    int frame_count = 0;
    do{
        display();
        glfwSwapBuffers(window);
        glfwPollEvents();
        ++frame_count;
        cur_time = glfwGetTime();
        if( cur_time - old_time > 1.0 )
        {
            showComputationTime(cur_time - old_time, frame_count);
            old_time = cur_time;
            frame_count = 0;
        }
        
        if(write_video_to_file){
            glReadPixels(0, 0, render_buffer_width, render_buffer_height, GL_RGB, GL_UNSIGNED_BYTE, frame.data );
            vid.write(frame);
        }
    }while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS && glfwWindowShouldClose(window) == 0 );
    
    if(write_video_to_file) vid.release();
    
    glfwTerminate();
    
    return 0;
}
