#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define NUM_ELEMENTS 1024
#define NUM_BINS 256

int main() {
    int *input = (int *)malloc(NUM_ELEMENTS * sizeof(int));
    int *bins = (int *)calloc(NUM_BINS, sizeof(int));
    struct timeval start, end;

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        input[i] = i % NUM_BINS;
    }

    gettimeofday(&start, NULL);

    #pragma omp parallel for
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        #pragma omp atomic
        bins[input[i]]++;
    }

    gettimeofday(&end, NULL);
    double duration = (end.tv_sec - start.tv_sec) * 1000.0;
    duration += (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Total execution time: %f ms\n", duration);

    free(input);
    free(bins);
    return 0;
}

