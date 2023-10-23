#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define M 3000
#define NUM_THREADS 512

int main(int argc, char * argv[]){

        int i,j,k,sum;
        double start, end;

        /** NO need to check if A_col matches B_rows because square matrix **/

        double (*A)[M] = malloc(M * M * sizeof(A[0][0]));
        double (*B)[M] = malloc(M * M * sizeof(B[0][0]));
        double (*C)[M] = malloc(M * M * sizeof(C[0][0])); // Result matrix C = A*B

        /** Loop_1: Initialize the A and B matrices (same loop bc square matrices) **/

        for(i = 0; i < M; i++) {
                for(j = 0; j< M; j++) {
                        A[i][j] = rand()%10;
                        B[i][j] = rand()%10;
                }
        }

        // Parallelization

        // Setting the total number of threads
        omp_set_num_threads(NUM_THREADS);

        int num_threads = omp_get_max_threads();

        start = omp_get_wtime();

        #pragma omp parallel shared(A,B,C, sum) private(i,j,k) 
        {

                #pragma omp for reduction(+ : sum)
                for (i = 0; i < M; ++i){
                        for (j =0; j<M; ++j){
                                sum = 0;
                                for (k = 0; k<M; ++k){
                                        sum += A[i][k] * B[k][j];
                                }
                                C[i][j] = sum;
                        }
                }
        }

        end = omp_get_wtime();

        double algo_time = (end - start);
        printf("\nNumber of threads: %d \t  - Elapsed (Execution time): %lf seconds\n", num_threads, algo_time);

        printf("\n\n");

        // Deallocate/Free the memory (No Memory leaks)

        free(A);
        A = NULL;
        free(B);
        B = NULL;
        free(C);
        C = NULL;

        return 0;
}
