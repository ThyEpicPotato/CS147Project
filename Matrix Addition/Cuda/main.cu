#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "kernel.cu"
#include "support.h"
#include <time.h>

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t mat_sz;
    unsigned matDim;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matDim = 1000;
    } else if (argc == 2) {
        matDim = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./mat-add                # All matrices are 1000 x 1000"
      "\n    Usage: ./mat-add <m>            # All matrices are m x m"
      "\n");
        exit(0);
    }
   
    mat_sz = matDim*matDim;

    A_h = (float*) malloc( sizeof(float)*mat_sz );
    for (unsigned int i=0; i < mat_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*mat_sz );
    for (unsigned int i=0; i < mat_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*mat_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matDim, matDim,
        matDim, matDim, matDim, matDim);
        
    /*
    std::cout << "A:";
    for (int i = 0; i < mat_sz; ++i) {
        std::cout << A_h[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << "B:";
    for (int i = 0; i < mat_sz; ++i) {
        std::cout << B_h[i] << ", ";
    }
    std::cout << std::endl;
    */

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    int size = mat_sz * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    /*************************************************************************/
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);
	
    /*************************************************************************/
    
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    /*************************************************************************/
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard mat-add interface -------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicMatAdd(matDim, A_d, B_d, C_d);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables to host ------------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    /*************************************************************************/
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------
    /*
    std::cout << "C:";
    for (int i = 0; i < mat_sz; ++i) {
        std::cout << C_h[i] << ", ";
    }
    std::cout << std::endl;
    */

    printf("Verifying results...\n"); fflush(stdout);

    verify(A_h, B_h, C_h, matDim);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    /*************************************************************************/
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    /*************************************************************************/
    return 0;
}

