//pgcc -acc -o matAddACC matAddACC.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

static int PRINT_FLAG = 0;
static int VERIFY_FLAG = 0;

void generate_random_matrix(int N, double matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
}

void add_matrices(int N, double A[N][N], double B[N][N], double C[N][N]) {
    #pragma acc parallel loop collapse(2) copyin(A[0:N][0:N], B[0:N][0:N]) copyout(C[0:N][0:N])
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void print_matrix(int N, double matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.2f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int verify_matrix_addition(int N, double matrixA[N][N], double matrixB[N][N], double matrixC[N][N]) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (matrixC[i][j] != matrixA[i][j] + matrixB[i][j]) {
                return 0; 
            }
        }
    }
    return 1;  
}

int main(int argc, char *argv[]) {
    int N = 1000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("%dx%d\n", N, N);

    srand(time(0));

    double (*A)[N] = malloc(sizeof(double[N][N]));
    double (*B)[N] = malloc(sizeof(double[N][N]));
    double (*C)[N] = malloc(sizeof(double[N][N]));

    generate_random_matrix(N, A);
    generate_random_matrix(N, B);

    add_matrices(N, A, B, C);

    if (PRINT_FLAG == 1) {
        printf("Matrix A:\n");
        print_matrix(N, A);

        printf("Matrix B:\n");
        print_matrix(N, B);

        printf("Matrix C (A + B):\n");
        print_matrix(N, C);
    }
    
    if (VERIFY_FLAG == 1) {
        if (verify_matrix_addition(N, A, B, C)) {
            printf("Matrix addition is correct.\n");
        } else {
            printf("Matrix addition is incorrect.\n");
        }
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
