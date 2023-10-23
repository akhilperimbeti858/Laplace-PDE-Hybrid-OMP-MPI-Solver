#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main( argc, argv )
int argc;
char **argv;
{
        int rank, value, size, i, j, itcnt;

        MPI_Status status;
        MPI_Datatype column;

        double     diffnorm, gdiffnorm;
        double     xlocal[7][7];
        double     xnew[7][7];
        double     result[12][12];

        MPI_Init( &argc, &argv );

        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );

        MPI_Barrier(MPI_COMM_WORLD); // to synchronize all 4 processes

        double start_time = MPI_Wtime();
        double avg_time;

        if (size != 4){MPI_Abort( MPI_COMM_WORLD, 1 );} // Only run using 4 processes

        int nb_rows = 7; //local array rows
        int nb_cols = 7; //local array columns

        int msg_tag1 = 1; // North-to-South Communication
        int msg_tag2 = 2; // South-to-North Communication
        int msg_tag3 = 3; // East-to-West Communication
        int msg_tag4 = 4; // West-to-East Communication


        // All values are set to the rank of the processor, except the adjacent values
        for(int i = 0; i<nb_rows; i++){
                for(int j = 0; j<nb_cols; j++) {
                        if ((i == 0) || (i == nb_rows-1)) {
                                xlocal[i][j] = -1;
                        }
                        else {
                                xlocal[i][j] = rank ;
                        }
                }
        }

        itcnt = 0; // number of iterations

        do {

                // Row Updates

                if (rank > 1) // If not in the first row of processors (if rank is 2 or 3)
                {
                        MPI_Send(&xlocal[1][0], nb_cols, MPI_DOUBLE, rank-2, msg_tag1, MPI_COMM_WORLD); // Send 2nd row of array to the processor above
                }
                if (rank < 2) //If not in the last row of processors (if rank is 0 or 1)
                {
                    MPI_Recv(&xlocal[nb_rows-1][0], nb_cols, MPI_DOUBLE, rank+2, msg_tag1, MPI_COMM_WORLD, &status); // Recieve message-tag 1 from below and update last row of array
                }
                if (rank < 2) // If not in the last row of processors (if rank is 0 or 1)
                {
                         MPI_Send(&xlocal[nb_rows-2][0], nb_cols, MPI_DOUBLE, rank+2, msg_tag2, MPI_COMM_WORLD); // Send 2nd-last row of array to the processor below
                }
                if (rank > 1) // If not in the first row of processors (if rank is 2 or 3)
                {
                        MPI_Recv(&xlocal[0][0], nb_cols, MPI_DOUBLE, rank-2, msg_tag2, MPI_COMM_WORLD, &status); // Recieve message-tag 2 from above and update first row of array
                }

                // Column Updates

                MPI_Type_vector(nb_rows, 1, nb_cols, MPI_DOUBLE, &column); //Creating Column vector type
                MPI_Type_commit(&column);

                if (rank % 2 != 1) // If not in the first column if processors (if rank is 0 or 2)
                {
                        MPI_Send(&xlocal[0][nb_cols-2], 1, column, rank+1, msg_tag3, MPI_COMM_WORLD); // Send 2nd-last column of array to the processor to the right (east-to-west)
                }
                if (rank % 2 != 0) // If not in the last column if processors (if rank is 1 or 3)
                {
                        MPI_Recv(&xlocal[0][0], 1, column, rank-1, msg_tag3, MPI_COMM_WORLD,  &status ); // Recieve message-tag 3 and update first column of array
                }
                if (rank % 2 != 0) // If not in the last column if processors (if rank is 1 or 3)
                {
                        MPI_Send(&xlocal[0][1], 1, column, rank-1, msg_tag4, MPI_COMM_WORLD); // Send 2nd column of array to the processor to the right (west-to-east)
                }
                if (rank % 2 != 1) // If not in the first column if processors (if rank is 0 or 2)
                {
                        MPI_Recv(&xlocal[0][nb_cols-1], 1, column, rank+1, msg_tag4, MPI_COMM_WORLD,  &status ); // Recieve message-tag 4 and update last column of array
                }


                // Compute new values and update using Jacobi iteration method
                itcnt++;
                diffnorm = 0.0;

                for (i=1; i< nb_rows-1; i++) {
                        for (j=1; j< nb_cols-1; j++) {

                                xnew[i][j] = (xlocal[i][j+1] + xlocal[i][j-1] +        // updating each point as avg. of 4 neighbor points
                                        xlocal[i+1][j] + xlocal[i-1][j]) / 4.0;

                                diffnorm += (xnew[i][j] - xlocal[i][j]) *           // Calculating the abs.error^2
                                        (xnew[i][j] - xlocal[i][j]);
                        }
                }

                /* Only transfer the interior points */
                for (i=1; i< nb_rows-1; i++) {
                        for (j=1; j<nb_cols-1; j++) {
                                xlocal[i][j] = xnew[i][j];
                        }
                }

                MPI_Allreduce( &diffnorm, &gdiffnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

                gdiffnorm = sqrt( gdiffnorm );

                if (rank == 0) { printf( "At iteration %d, diff is %e\n", itcnt, gdiffnorm ); }



        } while (gdiffnorm > 1.0e-2 && itcnt < 100); // Stopping Conditions
    
    
        double time = MPI_Wtime() - start_time;

        MPI_Reduce(&time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // average total time elapsed between all 4 processes




        if (rank == 0) {
            printf("--------------------------------------------------\n");
            printf("Total Time Elapsed (avg): %fs \n", avg_time/4);
            printf("--------------------------------------------------\n");
        }

        MPI_Finalize();

        return 0;

}
