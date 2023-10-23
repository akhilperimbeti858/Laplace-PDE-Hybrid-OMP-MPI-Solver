#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 10000000
#define pi_exact 3.14159265359
#define NUM_THREADS 64

int main(int argc, const char * argv[]){

        double x, y, dist, pi_approx;

        int i, count;

        // Creating an array with different seed values for the erand48 function

        unsigned short seed[5] = {0, 1, 2, 3,4};

        // Parellelization

        // Setting the total number of threads
        omp_set_num_threads(NUM_THREADS);

        int num_threads = omp_get_max_threads();

        #pragma omp parallel shared(count, seed) private(x,y,i)
        {
                #pragma omp for reduction(+: count)
                for (i=0; i<N; i++) {


                        x = (double) erand48(seed); // Random x-coord in [0.0, 1.0)
                        y = (double) erand48(seed); // Random y-coord in [0.0, 1.0)

                        // Calculating the sqaured distance of point (x,y) from the origin
                        dist = ((x*x) + (y*y));

                        // Now determining whether (x,y) falls inside the unit circle
                        if (dist <= 1) {
                                count += 1;

                        }
                }
        }

        // Calculating the approximate value of pi 

        pi_approx = 4.0 * ((double)count/(double)N);
        double err =  fabs(pi_exact - pi_approx);

        printf("\nTotal number of trials: %d, Number of threads: %d, Approximated Pi: %lf, Abs. Error: %f\n\n", N, num_threads, pi_approx, err);


        return 0;
}
