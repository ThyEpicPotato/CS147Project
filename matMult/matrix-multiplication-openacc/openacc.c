#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include <sys/time.h>

void matrixMultiply(float *a, float *b, float *c, int N){
#pragma acc data copyin(a[0:N*N], b[0:N*N]) copyout(c[0:N*N])
    {
        #pragma acc kernels
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += a[i * N + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    }
}


double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec * 1000 + (double)time.tv_usec * 0.001;
}

int main() {
    int sizes[] = {1500, 1750, 2000, 2250, 2500};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int idx = 0; idx < num_sizes; idx++) {
        int N = sizes[idx];
        float *a = (float *)malloc(N * N * sizeof(float));
        float *b = (float *)malloc(N * N * sizeof(float));
        float *c = (float *)malloc(N * N * sizeof(float));

        if (!a || !b || !c) {
            fprintf(stderr, "Memory allocation failed for size %d\n", N);
            return 1;
        }

        // Initialize matrices
                 for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++) {
                                             a[i * N + j] = (float)(i + j);
                                                             b[i * N + j] = (float)(i - j);
                                                                       }
                                                                                }
        
                                                                                         double start_time = get_wall_time();
                                                                                                 matrixMultiply(a, b, c, N);
                                                                                                         double end_time = get_wall_time();
        
                                                                                                                 printf("Data size: %d x %d\n", N, N);
                                                                                                                         printf("matrix_mul() time: %f msec\n", end_time - start_time);
        
                                                                                                                                 free(a);
                                                                                                                                         free(b);
                                                                                                                                                 free(c);
                                                                                                                                                     }
        
                                                                                                                                                         return 0;
                                                                                                                                                         }
        
