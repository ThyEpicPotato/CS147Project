// gcc -std=c99 -o CLDot CLDot.c -lOpenCL -lm
// ./CLDot 1000; default is 1000 for no arg
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DEFAULT_N 1000
#define CHECK_ERROR(err) if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL error: %d\n", err); exit(EXIT_FAILURE); }

const char *kernelSource = 
"__kernel void dotProductKernel(__global const float* a, __global const float* b, __global float* partialResults, int N) {\n"
"    __local float cache[256];\n"
"    int gid = get_global_id(0);\n"
"    int lid = get_local_id(0);\n"
"    int localSize = get_local_size(0);\n"
"    float temp = 0.0f;\n"
"    for (int i = gid; i < N; i += get_global_size(0)) {\n"
"        temp += a[i] * b[i];\n"
"    }\n"
"    cache[lid] = temp;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int i = localSize / 2; i > 0; i /= 2) {\n"
"        if (lid < i) {\n"
"            cache[lid] += cache[lid + i];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    if (lid == 0) {\n"
"        partialResults[get_group_id(0)] = cache[0];\n"
"    }\n"
"}\n"
"__kernel void reduceKernel(__global float* partialResults, int numGroups) {\n"
"    float sum = 0.0f;\n"
"    for (int i = 0; i < numGroups; i++) {\n"
"        sum += partialResults[i];\n"
"    }\n"
"    partialResults[0] = sum;\n"
"}\n";

void verifyResult(float* a, float* b, float result, int N) {
    float cpuResult = 0;
    for (int i = 0; i < N; ++i) {
        cpuResult += a[i] * b[i];
    }
    printf("CPU result: %.3f\n", cpuResult);
    printf("GPU result: %.3f\n", result);
    if (fabs(cpuResult - result) < 0.001)
        printf("Results are correct.\n");
    else
        printf("Results are incorrect.\n");
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;

    srand(time(0));

    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel dotProductKernel, reduceKernel;
    cl_mem d_a, d_b, d_partialResults;

    err = clGetPlatformIDs(1, &platform, NULL); CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK_ERROR(err);

    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err); CHECK_ERROR(err);

    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), h_a, &err); CHECK_ERROR(err);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), h_b, &err); CHECK_ERROR(err);
    int numGroups = (N + 255) / 256;
    d_partialResults = clCreateBuffer(context, CL_MEM_READ_WRITE, numGroups * sizeof(float), NULL, &err); CHECK_ERROR(err);

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err); CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "OpenCL build error: %s\n", buffer);
        exit(EXIT_FAILURE);
    }

    dotProductKernel = clCreateKernel(program, "dotProductKernel", &err); CHECK_ERROR(err);
    reduceKernel = clCreateKernel(program, "reduceKernel", &err); CHECK_ERROR(err);

    err = clSetKernelArg(dotProductKernel, 0, sizeof(cl_mem), &d_a); CHECK_ERROR(err);
    err = clSetKernelArg(dotProductKernel, 1, sizeof(cl_mem), &d_b); CHECK_ERROR(err);
    err = clSetKernelArg(dotProductKernel, 2, sizeof(cl_mem), &d_partialResults); CHECK_ERROR(err);
    err = clSetKernelArg(dotProductKernel, 3, sizeof(int), &N); CHECK_ERROR(err);

    // Execute dotProductKernel
    size_t localSize = 256;
    size_t globalSize = (N + localSize - 1) / localSize * localSize;
    cl_event dot_event, reduce_event;
    err = clEnqueueNDRangeKernel(queue, dotProductKernel, 1, NULL, &globalSize, &localSize, 0, NULL, &dot_event); CHECK_ERROR(err);

    err = clSetKernelArg(reduceKernel, 0, sizeof(cl_mem), &d_partialResults); CHECK_ERROR(err);
    err = clSetKernelArg(reduceKernel, 1, sizeof(int), &numGroups); CHECK_ERROR(err);

    // Execute reduceKernel
    err = clEnqueueTask(queue, reduceKernel, 0, NULL, &reduce_event); CHECK_ERROR(err);

    err = clFinish(queue); CHECK_ERROR(err);

    cl_ulong dot_start, dot_end;
    err = clGetEventProfilingInfo(dot_event, CL_PROFILING_COMMAND_START, sizeof(dot_start), &dot_start, NULL); CHECK_ERROR(err);
    err = clGetEventProfilingInfo(dot_event, CL_PROFILING_COMMAND_END, sizeof(dot_end), &dot_end, NULL); CHECK_ERROR(err);
    double dot_time = (dot_end - dot_start) / 1000000.0;
    printf("Time taken for dotProductKernel: %f ms\n", dot_time);

    cl_ulong reduce_start, reduce_end;
    err = clGetEventProfilingInfo(reduce_event, CL_PROFILING_COMMAND_START, sizeof(reduce_start), &reduce_start, NULL); CHECK_ERROR(err);
    err = clGetEventProfilingInfo(reduce_event, CL_PROFILING_COMMAND_END, sizeof(reduce_end), &reduce_end, NULL); CHECK_ERROR(err);
    double reduce_time = (reduce_end - reduce_start) / 1000000.0; // Convert to milliseconds
    printf("Time taken for reduceKernel: %f ms\n", reduce_time);

    printf("Total Time: %f ms\n", reduce_time + dot_time);

    float h_result;
    err = clEnqueueReadBuffer(queue, d_partialResults, CL_TRUE, 0, sizeof(float), &h_result, 0, NULL, NULL); CHECK_ERROR(err);

    verifyResult(h_a, h_b, h_result, N);

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_partialResults);
    clReleaseKernel(dotProductKernel);
    clReleaseKernel(reduceKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_a);
    free(h_b);

    return 0;
}
