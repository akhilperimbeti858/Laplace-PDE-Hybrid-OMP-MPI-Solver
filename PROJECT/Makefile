proj605_hyb:
	mpic++ -Wall -w hybrid_omp_mpi.c -o proj605_hyb -fopenmp -lm

run:
	mpiexec -np 16 -bind-to socket -map-by socket ./proj605_hyb

clean:
	rm proj605_hyb
