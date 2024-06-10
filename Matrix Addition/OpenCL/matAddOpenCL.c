// gcc -std=c99 -o matAddOpenCL matAddOpenCL.c -lOpenCL -lm
// ./matAddOpenCL N; default is 1000 for no arg
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>

static int PRINT_FLAG = 0;
static int VERIFY_FLAG = 0;

void generate_random_matrix(int N, float matrix[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
}

void print_matrix(int N, float matrix[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%8.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int verify_matrix_addition(int N, float matrixA[N][N], float matrixB[N][N], float matrixC[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (matrixC[i][j] != matrixA[i][j] + matrixB[i][j]) {
                return 0; 
            }
        }
    }
    return 1; 
}

const char *matrix_add_kernel =
    "__kernel void matrix_add(__global const float* A, __global const float* B, __global float* C, const int N) {"
    "    int i = get_global_id(0);"
    "    int j = get_global_id(1);"
    "    int index = i * N + j;"
    "    C[index] = A[index] + B[index];"
    "}";

void check_error(cl_int err, const char *operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    int N = 1000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    printf("%dx%d\n", N, N);

    srand(time(0));

    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));

    generate_random_matrix(N, A);
    generate_random_matrix(N, B);

    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufA, bufB, bufC;

    size_t global_size[2] = {N, N};
    size_t local_size[2] = {16, 16};
    size_t max_local_size;

    cl_int err;

    err = clGetPlatformIDs(1, &platform_id, NULL);
    check_error(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
    check_error(err, "clGetDeviceIDs");

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    check_error(err, "clCreateContext");

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    command_queue = clCreateCommandQueueWithProperties(context, device_id, properties, &err);
    check_error(err, "clCreateCommandQueueWithProperties");

    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer(bufA)");
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * N * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer(bufB)");
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer(bufC)");

    err = clEnqueueWriteBuffer(command_queue, bufA, CL_TRUE, 0, N * N * sizeof(float), A, 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer(bufA)");
    err = clEnqueueWriteBuffer(command_queue, bufB, CL_TRUE, 0, N * N * sizeof(float), B, 0, NULL, NULL);
    check_error(err, "clEnqueueWriteBuffer(bufB)");

    program = clCreateProgramWithSource(context, 1, &matrix_add_kernel, NULL, &err);
    check_error(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Error in kernel: %s\n", log);
        free(log);
        exit(1);
    }

    kernel = clCreateKernel(program, "matrix_add", &err);
    check_error(err, "clCreateKernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    check_error(err, "clSetKernelArg(bufA)");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    check_error(err, "clSetKernelArg(bufB)");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    check_error(err, "clSetKernelArg(bufC)");
    err = clSetKernelArg(kernel, 3, sizeof(int), &N);
    check_error(err, "clSetKernelArg(N)");

    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_local_size, NULL);
    check_error(err, "clGetKernelWorkGroupInfo");

    if (local_size[0] * local_size[1] > max_local_size) {
        local_size[0] = local_size[1] = (size_t) sqrt(max_local_size);
    }

    global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
    global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

    // Execute
    cl_event event;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &event);
    check_error(err, "clEnqueueNDRangeKernel");

    clFinish(command_queue);

    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double time_spent = (time_end - time_start) / 1000000.0;
    printf("Time taken to add matrices: %f ms\n", time_spent);

    err = clEnqueueReadBuffer(command_queue, bufC, CL_TRUE, 0, N * N * sizeof(float), C, 0, NULL, NULL);
    check_error(err, "clEnqueueReadBuffer(bufC)");

    if (PRINT_FLAG == 1) {
        printf("Matrix A:\n");
        print_matrix(N, A);

        printf("Matrix B:\n");
        print_matrix(N, B);

        printf("Matrix C (A + B):\n");
        print_matrix(N, C);
    }

    if (VERIFY_FLAG == 1) {
        if (verify_matrix_addition(N, A, B, C)) {
            printf("Matrix addition is correct.\n");
        } else {
            printf("Matrix addition is incorrect.\n");
        }
    }

    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);

    return 0;
}
