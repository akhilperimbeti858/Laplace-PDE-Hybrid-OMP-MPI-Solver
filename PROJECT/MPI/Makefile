proj605_mpi:
	mpic++ -Wall -w purempi.c -o proj605_mpi -lm

run:
	mpirun -np 16 -bind-to socket -map-by socket ./proj605_mpi

clean:
	rm proj605_mpi
