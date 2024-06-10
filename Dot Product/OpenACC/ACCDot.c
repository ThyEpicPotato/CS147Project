// pgcc -acc -o ACCDot ACCDot.c
// ./ACCDot 1000; default is 1000 for no arg
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <openacc.h>

#define DEFAULT_N 1000

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

    float result = 0;

    clock_t start = clock();

    #pragma acc data copyin(h_a[0:N], h_b[0:N]) copy(result)
    {
        #pragma acc parallel loop reduction(+:result)
        for (int i = 0; i < N; ++i) {
            result += h_a[i] * h_b[i];
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("Time taken to add matrices: %f ms\n", time_spent);

    verifyResult(h_a, h_b, result, N);

    free(h_a);
    free(h_b);

    return 0;
}
