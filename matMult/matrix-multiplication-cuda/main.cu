#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

int main() {
    Timer timer;
    cudaError_t cuda_ret;
    int sizes[] = {1500, 1750, 2000};  // Example sizes
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int idx = 0; idx < numSizes; ++idx) {
        int N = sizes[idx];
        float *A_h, *B_h, *C_h;
        float *A_d, *B_d, *C_d;
        size_t numElements = N * N;
        size_t size = numElements * sizeof(float);


       A_h = (float*) malloc(size);
        B_h = (float*) malloc(size);
        C_h = (float*) malloc(size);

       for (int i = 0; i < numElements; i++) {
            A_h[i] = (rand() % 100) / 100.0f;
            B_h[i] = (rand() % 100) / 100.0f;
        }

        cudaMalloc((void**)&A_d, size);
        cudaMalloc((void**)&B_d, size);
        cudaMalloc((void**)&C_d, size);

       cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

        printf("Running matrix multiplication for size %dx%d\n", N, N);
        startTime(&timer);

        dim3 dimBlock(16, 16);  // Block dimensions
        dim3 dimGrid((N + 15) / 16, (N + 15) / 16);  // Grid dimensions

        cuda_ret = cudaDeviceSynchronize();
        stopTime(&timer);
        if(cuda_ret != cudaSuccess) printf("Kernel execution failed: %s\n", cudaGetErrorString(cuda_ret));
        else printf("Elapsed time: %f ms\n", elapsedTime(timer));

        cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

        verify(A_h, B_h, C_h, N, N, N);

        free(A_h); free(B_h); free(C_h);
        cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    }
    return 0;
}
