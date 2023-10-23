#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define pi_exact 3.14159265359
#define N 10000000
#define NUM_THREADS 16

double f(double x);

int main(int argc, const char * argv[]){

        double a = 0.0 , b = 1.0;
        double integral, err, start, end;
        int i;

        double h = (b - a)/N;

        integral = (f(a) + f(b))/2.0;

        //Parellelization 
        omp_set_num_threads(NUM_THREADS);

        int num_threads = omp_get_max_threads();

        start = omp_get_wtime();

        #pragma omp parallel shared (a,b ,h,integral) private (i) 
        {
                #pragma omp parallel for reduction(+:integral)
                for (i = 0; i <N; i++){
                        integral = integral + f(a+(i*h));
                }
        }

        end = omp_get_wtime();

        double wtime = (end-start);
        double total = integral*h;

        err = fabs(total - pi_exact);

        printf ( "\n" );
        printf ( "  Estimate = %24.16f\n", total );
        printf ( "  Error    = %e\n", err );

        printf ( "  N  = %d\n", N );
        printf ( "  Total # of threads   = %d\n", num_threads );
        printf ( "  Time     = %f\n", wtime );

        return 0;
}

double f(double x){
        return ((4.0)/(1+(x*x)));

}
