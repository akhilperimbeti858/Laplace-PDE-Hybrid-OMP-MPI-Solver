#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "mpi.h"
#include "omp.h"


#define N 1280 // Num rows
#define M 640 // Num cols

int main(int argc, char* argv[])
{
    int        rank, size, i, j, itcnt;
    int        nb_cols, nb_rows;
    int        nprocs, max_iters;
    double     diffnorm, gdiffnorm;               // for storing error
    double     start_time;                     // variable for timing purposes



    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    MPI_Status status;

    nprocs = size;

    int msg_tag1 = 1; // North-to-South Communication
    int msg_tag2 = 2; // South-to-North Communication
    int msg_tag3 = 3; // West-to-East Communication
    int msg_tag4 = 4; // East-to-West Communication


    // ERROR EXCEPTIONS - trigger MPI_Abort() if any of these conditions are met

    if ((M*N) % size != 0 || size < 2) {
        printf("\n ERROR: The domain size must be evenly divided by the total number of processors. Minimum amount of processors: 2 \n");
        MPI_Abort( MPI_COMM_WORLD, 1 ); // If the dimensions or processors do not allow for an even number of 'blocks'
    }

    if (size%2 == 1 || (M*N)%2 == 1 ){
        printf("\n ERROR: Number of processors must be even. M and N must be even \n");
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }

    nb_cols = (M/2)+1;  // Column dimension for local array
    nb_rows = (N/(nprocs/2))+1; //Row dimension for local array

    if (size > 4){ // More than 4 processors
                // When process is not one of the edge 'blocks'
            if ((rank !=0) && (rank != nprocs/2) && (rank != ((nprocs/2)-1)) && (rank != (nprocs - 1))){
                    nb_cols = nb_cols + 1;
            }
    }

    if (size == 2) { // if exactly 2 processors
            nb_cols = nb_cols - 1;
    }

    /* Initializing the local arrays with Initial and Boundary Conditions */

    double *xlocal = new double[nb_rows*nb_cols];
    double *xnew = new double[nb_rows*nb_cols];

    // Values are set to the rank of the processor, except the adjacent values

    // Upper and Lower boundaries are set to -1

    for(int i = 0; i<nb_rows; i++) {
        for(int j = 0; j<nb_cols; j++) {
            if ((i == 0) || (i == nb_rows-1)) {

                xlocal[i*nb_cols + j] = -1;
            }
            else {

                xlocal[i*nb_cols + j] = rank;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();  // start the timer once all processes have reached this point

    itcnt = 0;
    do {

        /* COLUMN COMMUNICATION - Sending Columns left or right */

        // Declare MPI Column vector type to exchange columns between processes
        MPI_Datatype column;
        MPI_Type_vector(nb_rows, 1, nb_cols, MPI_DOUBLE, &column);
        MPI_Type_commit(&column);


        if ((rank % 2 != 1)) { // If not in the first column of processors (if rank is an EVEN value)

                // Send the last column of array to the processor to the right (west-to-east)
                MPI_Send(&xlocal[0*nb_cols + (nb_cols-2)],  1, column, rank + 1, msg_tag3, MPI_COMM_WORLD );
        }

        if ((rank % 2 != 0)) { // If not in the last column of processors (if rank is an ODD value)

                // Recieve message-tag 3 and update first column of array
                MPI_Recv(&xlocal[0*nb_cols + 0],  1, column, rank - 1, msg_tag3, MPI_COMM_WORLD, &status);
        }

        if ((rank %2 != 0)) { // If not in the last column of processors (if rank is an ODD value)

            // Send the first column of array to the processor to the left (east-to-west)
            MPI_Send(&xlocal[0*nb_cols + 1], 1, column, rank - 1,  msg_tag4, MPI_COMM_WORLD );
        }

        if((rank % 2 != 1 )) { // If not in the first column of processors (if rank is an EVEN value)

                // Recieve message-tag 4 and update last column of array
                MPI_Recv(&xlocal[0*nb_cols + (nb_cols-1)],  1, column, rank + 1, msg_tag4, MPI_COMM_WORLD, &status );
        }



        /* ROW COMMUNICATION - Sending Rows up or down */

        if (rank < (nprocs/2)) {   // If not in the bottom half of processors (if rank < (number of processes)/2 )

            // Send the last row of array to the processor below (north-to-south)
            MPI_Send(&xlocal[(nb_rows-2)*nb_cols + 0], nb_cols, MPI_DOUBLE, rank + (nprocs/2), msg_tag1, MPI_COMM_WORLD );
        }

        if (rank > (nprocs/2) - 1) {    // If not in the bottom half of processors (if rank >= (number of processes)/2)

            // Recieve message-tag 2 from above and update first row of array
            MPI_Recv(&xlocal[0*nb_cols + 0], nb_cols, MPI_DOUBLE, rank - (nprocs/2), msg_tag1, MPI_COMM_WORLD, &status );
        }

        if (rank > (nprocs/2) -1 ) { // If not in the bottom half of processors (if rank >= (number of processes)/2)

            // Send first row of array to the processor above (south-to-north)
            MPI_Send(&xlocal[1*nb_cols + 0], nb_cols, MPI_DOUBLE, rank - (nprocs/2), msg_tag2, MPI_COMM_WORLD );
        }

        if (rank < (nprocs/2)) {  // If not in the bottom half of processors (if rank < (number of processes)/2 )

            // Recieve message-tag 1 from below and update last row of array
            MPI_Recv(&xlocal[(nb_rows-1) * nb_cols + 0], nb_cols, MPI_DOUBLE, rank + (nprocs/2), msg_tag2, MPI_COMM_WORLD, &status );
        }



        /* JACOBI Iteration! - Pure MPI - No OpenMP implementation*/

        itcnt ++;
        diffnorm = 0.0;  //normalized error value calculated as difference in solution between iterations

        // iterating only over the interior points- Jacobi iteration calculated as avg of 4 neighbouring values

        for (i=1; i<=nb_rows-2; i++) {
            for (j=1; j<=nb_cols-2; j++) {
                xnew[i*nb_cols + j] = (xlocal[i*nb_cols + (j+1)] + xlocal[i*nb_cols + (j-1)] +
                              xlocal[(i+1)*nb_cols + j] + xlocal[(i-1)*nb_cols + j]) / 4.0;

                diffnorm += (xnew[i*nb_cols + j] - xlocal[i*nb_cols + j]) *
                            (xnew[i*nb_cols + j] - xlocal[i*nb_cols + j]);
            }
         }

        /* Only transfer the interior points */

            for (i=1; i<=nb_rows-2; i++) {
                for (j=1; j<=nb_cols-2; j++) {
                xlocal[i*nb_cols + j] = xnew[i*nb_cols + j];          // Saves solution for process only at the interior points
                }
            }


        MPI_Allreduce( &diffnorm, &gdiffnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD ); // summing error from all processes

        gdiffnorm = sqrt( gdiffnorm );              // calculate error

        MPI_Allreduce(&itcnt, &max_iters, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // Finding total number (max value) of iterations

    } while (gdiffnorm > 1.0e-2); //Stopping condition if gloabal error falls below tolerance threshold

    MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes to finish

    double time = MPI_Wtime() - start_time;
    double total_time;    // total time for all threads

    MPI_Reduce(&time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // Summing the time elapsed for each process for the total


    if (rank == 0) {
        printf("\n           Pure-MPI Implementation     \n" );
        printf("--------------------------------------------------\n");
        printf("                    SUMMARY:                 \n");
        printf("--------------------------------------------------\n");
        printf("      GRID SIZE (rows x cols): %d x %d \n", N, M);
        printf("--------------------------------------------------\n");
        printf("        Number of Processes (MPI): %d \n", nprocs);
        printf("--------------------------------------------------\n");
        printf("                   RESULTS:                \n");
        printf("--------------------------------------------------\n");
        printf("      Total Number of Iterations : %d \n", max_iters);
        printf("--------------------------------------------------\n");
        printf("         Time Elapsed (avg.) : %fs \n", (total_time/nprocs));
        printf("--------------------------------------------------\n\n");
    }



    // Free the column datatype
    int MPI_Type_free(MPI_Datatype column);

    // Free all allocated memory for local arrays - set pointers to NULL

    delete[] xlocal;
    xlocal = nullptr;

    delete[] xnew;
    xnew = nullptr;

    MPI_Finalize( );

    return 0;
}

