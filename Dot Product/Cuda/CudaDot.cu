// nvcc -o CudaDot CudaDot.cu
// ./CudaDot 1000; default is 1000 for no arg
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <time.h>

#define DEFAULT_N 1000

__global__ void dotProductKernel(float* a, float* b, float* result, int N) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    cache[cacheIndex] = temp;
    
    __syncthreads();
    
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    
    if (cacheIndex == 0) {
        atomicAdd(result, cache[0]);
    }
}

void verifyResult(float* a, float* b, float result, int N) {
    float cpuResult = 0;
    for (int i = 0; i < N; ++i) {
        cpuResult += a[i] * b[i];
    }
    std::cout << "CPU result: " << cpuResult << "\n";
    std::cout << "GPU result: " << result << "\n";
    if (fabs(cpuResult - result) < 0.001) {
        std::cout << "Results are correct.\n";
    } else {
        std::cout << "Results are incorrect.\n";
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;

    srand(time(0));

    float* h_a = new float[N];
    float* h_b = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_a, *d_b, *d_result;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    float h_result = 0;
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    clock_t start = clock();

    dotProductKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, N);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    std::cout << "Time taken to add matrices: " << time_spent << "ms" << std::endl;

    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    verifyResult(h_a, h_b, h_result, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    delete[] h_a;
    delete[] h_b;

    return 0;
}
