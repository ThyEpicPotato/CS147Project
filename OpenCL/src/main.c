#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

const char *kernelSource = 
"__kernel void histogram_kernel(__global const int *input, __global int *bins, const int num_elements) {\n"
"    int idx = get_global_id(0);\n"
"    if (idx < num_elements) {\n"
"        atomic_inc(bins + input[idx]);\n"
"    }\n"
"}\n";

int main() {
    int num_elements = 1024;
    int num_bins = 256;
    int *input = (int *)malloc(num_elements * sizeof(int));
    int *bins = (int *)calloc(num_bins, sizeof(int));
    for (int i = 0; i < num_elements; i++) {
        input[i] = i % num_bins;
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "histogram_kernel", &err);

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, num_elements * sizeof(int), NULL, &err);
    cl_mem bins_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_bins * sizeof(int), NULL, &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bins_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &num_elements);
    clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, num_elements * sizeof(int), input, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bins_buffer, CL_TRUE, 0, num_bins * sizeof(int), bins, 0, NULL, NULL);

    size_t global_size = num_elements;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, bins_buffer, CL_TRUE, 0, num_bins * sizeof(int), bins, 0, NULL, NULL);

    clFinish(queue);

    gettimeofday(&end, NULL);
    double duration = (end.tv_sec - start.tv_sec) * 1000.0;
    duration += (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Total execution time: %f ms\n", duration);

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(bins_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    free(input);
    free(bins);
    return 0;
}

