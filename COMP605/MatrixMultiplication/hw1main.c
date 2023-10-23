//
//  main.c
//  COMP605_HW1
//
//  Created by Akhil on 2/9/22.
//

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, const char * argv[]) {
    
    // Declaring and Initializing variables
    int A_ROWS, A_COLUMNS, B_ROWS, B_COLUMNS;
    
    struct timeval start, end;
    
    int i,j,k,row, col, sum, r;
    
    // For printing and storing result matrix C
    char * output_path;
    FILE * Result_C;
    
    printf("\n-----------------------------------------------------------------------------------\n");
    printf("\t\t\tCOMP_605 HW1, Author: Akhil Perimbeti\n");
    printf("-----------------------------------------------------------------------------------\n");
    printf("This program performs a matrix-multiplication between Matrix A and Matrix B (A * B). \nIt compares the execution time between the ijk-form and the jki-form at two different \nGNU compiler optimization options -> -o0 (no opimization) & -o3 (full optimization). \n");
    printf("------------------------------------------------------------------------------------\n");
    
    printf("\nEnter the number of rows for Matrix A: ");
    scanf("%d", &A_ROWS);
    printf("Enter the number of columns for Matrix A: ");
    scanf("%d", &A_COLUMNS);
    printf("\nEnter the number of rows for Matrix B: ");
    scanf("%d", &B_ROWS);
    printf("Enter the number of columns for Matrix B: ");
    scanf("%d", &B_COLUMNS);
    
    printf("\nSize of A: %d x %d, Size of B: %d x %d matrix --> Size of C = (A*B): %d x %d", A_ROWS, A_COLUMNS, B_ROWS, B_COLUMNS, A_ROWS, B_COLUMNS);
    printf("\n------------------------------------------------------------------------------------\n\n");
    

    // Checking if Columns of Matrix A match the Rows of Matrix B
    if (A_COLUMNS != B_ROWS){
        char str1[] = "ERROR: Columns of A and Rows of B must be equal for Matrix-Multiplication \n";
        printf("%s", str1);
        exit(EXIT_FAILURE);
    }
    
    // Allocating memory for matrix A, B and result matrix C1 and C2
    // The result matrix C would be of size: [A_ROWS] x [B_COLUMNS]
    
    int (*A)[A_COLUMNS] = malloc(A_ROWS * A_COLUMNS * sizeof(A[0][0]));
    int (*B)[B_COLUMNS] = malloc(A_ROWS * A_COLUMNS * sizeof(B[0][0]));
    int (*C)[B_COLUMNS] = malloc(A_ROWS * B_COLUMNS * sizeof(C[0][0])); // Result matrix C = (A*B)
    
    // Filling Matrix A with random integers between 0-9
    //srand(time(NULL));
    for(i = 0; i < A_ROWS; i++){
        for(j = 0; j< A_COLUMNS; j++){
            A[i][j] = rand()%10;
        }
    }
    
    // Filling Matrix B with random integers between 0-9
    //srand(time(NULL));
    for(i = 0; i < B_ROWS; i++){
        for(j = 0; j< B_COLUMNS; j++){
            B[i][j] = rand()%10;
        }
    }
    

    // ALGORITHM 1 - ijk-form (Results stored in matrix C)
    
    gettimeofday(&start, NULL);

    for (i = 0; i < A_ROWS; ++i){
        for (j = 0; j < B_COLUMNS; ++j){
            sum = 0;
            for (k = 0;  k < A_COLUMNS; ++k){
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    gettimeofday(&end, NULL);
    
    double algo1_time = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec))/1000000;
    printf("Algorithm 1 (ijk-form) - Elapsed (Execution time): %lf seconds\n", algo1_time);
    
    
    
    // Resetting the values of Matrix C for Algorithm 2
    for (row = 0; row < A_ROWS; row++){
        for (col = 0; col < B_COLUMNS; col++){
            C[row][col] = 0;
        }
    }
    
    
    // ALGORITHM 2 - jki-form (Results stored in matrix C2)
    
    gettimeofday(&start, NULL);
    
    for (j = 0; j < B_COLUMNS; ++j){
        for (k = 0; k < A_COLUMNS; ++k){
            r = B[k][j];
            for (i = 0; i < A_ROWS; ++i){
                C[i][j] += A[i][k] * r;
            }
        }
    }
    
    gettimeofday(&end, NULL);
    
    double algo2_time = ((end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec)/1000000.0;
    printf("\nAlgorithm 2 (jki-form) - Elapsed (Execution time): %lf seconds\n", algo2_time);
        
    
    printf("\n------------------------------------------------------------------------------------\n");
    printf("The results matrix C = (A*B) of size (%d x %d) is in the output file: Result_C.txt", A_ROWS, B_COLUMNS);
    
    
    // Printing Matrix C to output file: Results_C.txt
    
    output_path = "Result_C.txt";
    Result_C = fopen(output_path, "w+"); // open output file
    
    if (!Result_C) {
        perror(output_path); // Print error message to stdout - Cant open Result_C
        exit(EXIT_FAILURE);
    }
    //Write the results into the output files
    for(row = 0; row < A_ROWS; row++){
        for(col = 0; col < B_COLUMNS; col++){
            if (col == B_COLUMNS-1){
            fprintf(Result_C, " %d\n", C[row][col]);
            }
            else {
            fprintf(Result_C, " %d", C[row][col]);
            }
        }
    }
    fclose(Result_C); // Close output file
    
    
    // Deallocate/Free the memory (No Memory leaks)

    free(A);
    A = NULL;
    free(B);
    B = NULL;
    free(C);
    C = NULL;

    printf("\n-------------------------------------------------------------------------------------\n\n");
    return 0;
}

