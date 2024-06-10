#include <stdio.h>

#define TILE_SIZE 16

__global__ void matAdd(int dim, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (dim x dim) matrix
     *   where B is a (dim x dim) matrix
     *   where C is a (dim x dim) matrix
     *
     ********************************************************************/

    /*************************************************************************/
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < (dim * dim)) {
        C[i] = A[i] + B[i];
        /*
        printf("Element A%d: %f\n", i, A[i]);
        printf("Element B%d: %f\n", i, B[i]);
        printf("Element C%d: %f\n", i, C[i]);
        */
    }
    /*************************************************************************/

}

void basicMatAdd(int dim, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    /*
    int size = (dim * dim) * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **) &d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_C, size);
    */

    /*************************************************************************/
	dim3 DimGrid(((dim * dim)-1)/256 + 1, 1, 1);
    dim3 DimBlock(256, 1, 1);
	matAdd<<<DimGrid,DimBlock>>>(dim, A, B, C);

    /*************************************************************************/
    /*
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
	*/
    /*************************************************************************/

}

