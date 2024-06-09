#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>

double get_current_time() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    return te.tv_sec + te.tv_usec * 1e-6;
}

const char *kernelSource = "\
__kernel void matrixMultiply(__global float* A, __global float* B, __global float* C, int M, int N, int K) {\
    int row = get_global_id(0);\
    int col = get_global_id(1);\
    float sum = 0.0;\
    for(int i = 0; i < K; i++) {\
        sum += A[row * K + i] * B[i * N + col];\
    }\
    C[row * N + col] = sum;\
}\n";

int main() {
    int sizes[] = {1500, 1750, 2000}; // Different matrix sizes for testing
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

 clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matrixMultiply", &err);

    for (int idx = 0; idx < num_sizes; idx++) {
        int N = sizes[idx];
        float *A = (float*)malloc(sizeof(float) * N * N);
        float *B = (float*)malloc(sizeof(float) * N * N);
        float *C = (float*)malloc(sizeof(float) * N * N);

   for(int i = 0; i < N * N; i++) {
            A[i] = i % 100;
            B[i] = i % 100;
        }

        cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
        cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
        cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &err);

        clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, N * N * sizeof(float), A, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, N * N * sizeof(float), B, 0, NULL, NULL);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
        clSetKernelArg(kernel, 3, sizeof(int), &N);
        clSetKernelArg(kernel, 4, sizeof(int), &N);
        clSetKernelArg(kernel, 5, sizeof(int), &N);

    size_t global[2] = { N, N }; // global domain size for our calculation
        size_t local[2] = { 16, 16 }; // local domain size for our calculation

       double start_time = get_current_time();
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
        clFinish(queue);
        double end_time = get_current_time();

        printf("Matrix size: %dx%d, Execution time: %f seconds\n", N, N, end_time - start_time);

        clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, N * N * sizeof(float), C, 0, NULL, NULL);

        free(A);
        free(B);
        free(C);
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}


