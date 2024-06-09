// File: CUDA/src/main.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void histogram_kernel(int *input, int *bins, int num_elements, int num_bins) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elements) {
        atomicAdd(&bins[input[idx]], 1);
    }
}

int main() {
    int num_elements = 1024; // Total number of input elements
    int num_bins = 256;      // Number of bins
    int *input, *bins;
    int *d_input, *d_bins;

    // Allocate host memory
    input = (int *)malloc(num_elements * sizeof(int));
    bins = (int *)calloc(num_bins, sizeof(int));

    // Initialize input with some values (simple pattern for testing)
    for (int i = 0; i < num_elements; i++) {
        input[i] = i % num_bins;
    }

    // Allocate device memory
    cudaMalloc(&d_input, num_elements * sizeof(int));
    cudaMalloc(&d_bins, num_bins * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, input, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_bins, 0, num_bins * sizeof(int));

    // Launch kernel
    histogram_kernel<<<(num_elements + 255) / 256, 256>>>(d_input, d_bins, num_elements, num_bins);

    // Copy results back to host
    cudaMemcpy(bins, d_bins, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_bins);
    free(input);
    free(bins);

    return 0;
}

